from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Union
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy
from sdlm.data.preprocessors import t5_random_spans_mask_batch, insert_extra_paddings, gpt_span_mask_batch, Objective
import torch
import numpy as np
import pdb
from random import choices


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

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        padding: Union[bool, str, PaddingStrategy] = True,
        max_length: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: str = "pt",
        span_infilling: bool = False,
        mixed_pretrain_objectives: bool = False,
        mask_ratio: float = 0.15,
        mean_mask_span_length: int = 3,
        extra_padding_ratio=0.0,
        seed: int = 42,
    ):
        self.tokenizer = tokenizer
        self.padding = padding
        self.max_length = max_length
        self.pad_to_multiple_of = pad_to_multiple_of
        self.return_tensors = return_tensors
        self.span_infilling = span_infilling
        self.extra_padding_ratio = extra_padding_ratio
        self.rng = np.random.default_rng(seed)
        self.mixed_pretrain_objectives = mixed_pretrain_objectives
        if self.mixed_pretrain_objectives:
            self.mask_generator = {}
            self.mask_generator[Objective.t5] = lambda batch: t5_random_spans_mask_batch(
                batch, mask_ratio=0.15, mean_mask_span_length=3, rng=self.rng
            )
            self.mask_generator[Objective.aggressive_t5] = lambda batch: t5_random_spans_mask_batch(
                batch, mask_ratio=0.5, mean_mask_span_length=8, rng=self.rng
            )
            self.mask_generator[Objective.prefix] = lambda batch: gpt_span_mask_batch(batch)
            self.mask_generator[Objective.unconditional] = lambda batch: None
        elif self.span_infilling:
            self.mask_generator = lambda batch: t5_random_spans_mask_batch(
                batch, mask_ratio, mean_mask_span_length, self.rng
            )

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        [f.pop("attention_mask") for f in features]

        if self.extra_padding_ratio:
            # Inserting random tokens uniformly, we do not modify start and end of
            # sequence tokens.
            for i in range(len(features)):
                features[i]["input_ids"] = insert_extra_paddings(
                    self.rng, features[i]["input_ids"], self.tokenizer.pad_token_id, self.extra_padding_ratio
                )

        masks = {}
        if self.span_infilling:
            # Generates masks and pads them.
            masks = {"span_mask": self.mask_generator(features)}
        elif self.mixed_pretrain_objectives:
            objectives = [Objective.unconditional, Objective.t5, Objective.prefix, Objective.aggressive_t5]
            weights = [0.25, 0.25, 0.25, 0.25]
            objective = choices(objectives, weights)[0]
            masks = {"span_mask": self.mask_generator[objective](features)}

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
