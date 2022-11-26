from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Union
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy
from sdlm.data.preprocessors import t5_random_spans_mask, insert_extra_paddings
import torch
import numpy as np 
import pdb 

@dataclass
class SpanInfillingDataCollator:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            - `True` or `'longest'` (default): Pad to the longest sequence in the batch (or no padding if only a single
              sequence is provided).
            - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
              acceptable input length for the model if that argument is not provided.
            - `False` or `'do_not_pad'`: No padding (i.e., can output a batch with sequences of different lengths).
        max_length (`int`, *optional*):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        return_tensors (`str`):
            The type of Tensor to return. Allowable values are "np", "pt" and "tf".
    """
    def __init__(self, tokenizer: PreTrainedTokenizerBase, 
                padding: Union[bool, str, PaddingStrategy] = True,
                max_length: Optional[int] = None,
                pad_to_multiple_of: Optional[int] = None,
                return_tensors: str = "pt",
                span_infilling: bool = False,
                mask_ratio: float = 0.15,
                mean_mask_span_length: int = 3,
                extra_padding_ratio = 0.0,
                seed: int = 42):
        self.tokenizer = tokenizer 
        self.padding = padding 
        self.max_length = max_length 
        self.pad_to_multiple_of = pad_to_multiple_of
        self.return_tensors = return_tensors
        self.span_infilling = span_infilling
        self.extra_padding_ratio = extra_padding_ratio
        self.rng = np.random.default_rng(seed)
        self.mask_generator = lambda length, pad_length : t5_random_spans_mask(length, mask_ratio, mean_mask_span_length, self.rng, pad_length)

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        [f.pop("attention_mask") for f in features]
            
        if self.extra_padding_ratio:
            # Inserting random tokens uniformly, we do not modify start and end of 
            # sequence tokens.
            for i in range(len(features)):
                features[i]['input_ids'] = insert_extra_paddings(
                self.rng, 
                features[i]["input_ids"], 
                self.tokenizer.pad_token_id, 
                self.extra_padding_ratio)

        masks = {}
        if self.span_infilling:
            # Generates masks and pads them.
            lengths = [len(feature["input_ids"]) for feature in features]
            max_length = max(lengths)
            masks = [self.mask_generator(length, max_length-length) for length in lengths]
            masks = {"span_mask": torch.tensor(masks)}
            
        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        if "label" in batch:
            batch["labels"] = batch["label"]
            del batch["label"]
        if "label_ids" in batch:
            batch["labels"] = batch["label_ids"]
            del batch["label_ids"]
        return {**batch, **masks}
