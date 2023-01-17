from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Union
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy
from sdlm.data.preprocessors import t5_random_spans_mask_batch, insert_extra_paddings, gpt_span_mask_batch
import numpy as np
import pdb
from random import choices
import torch
from enum import Enum
import random

class Objective(Enum):
    # Prefix language modeling like GPT style pretraining.
    prefix = 1
    # T5 objective with a range of 2 to 5 tokens as the span length, which masks about 15% of input tokens.
    t5 = 2
    # Aggressive denoising where approximately 50% of the input sequence is masked.
    aggressive_t5 = 3
    # Unconditional generation case.
    unconditional = 4


# TODO: automize this one.
# TODO: these are for sequence length of 100, adapt for 200.
OBJECTIVE_SETTINGS = {
    Objective.t5: [
        {"mask_ratio": 0.15, "mean_mask_span_length": 8},
        {"mask_ratio": 0.15, "mean_mask_span_length": 3},
    ],
    Objective.aggressive_t5: [
        {"mask_ratio": 0.5, "mean_mask_span_length": 8},
        {"mask_ratio": 0.5, "mean_mask_span_length": 3},
        {"mask_ratio": 0.5, "mean_mask_span_length": 48},
    ],
}


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
        mode,
        data_args,
        tokenizer: PreTrainedTokenizerBase,
        padding: Union[bool, str, PaddingStrategy] = True,
        max_length: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: str = "pt",
        seed: int = 42,
        eval_context_size: int = None,
    ):
        self.tokenizer = tokenizer
        self.padding = padding
        self.max_length = max_length
        self.pad_to_multiple_of = pad_to_multiple_of
        self.return_tensors = return_tensors
        self.conditional_generation = data_args.conditional_generation
        self.extra_padding_ratio = data_args.extra_padding_ratio
        self.rng = np.random.default_rng(seed)
        self.eval_context_size = eval_context_size
        self.mode = mode
        if self.conditional_generation == "ul2_with_unconditional" and mode == "train":
            self.mask_generator = {}
            self.mask_generator[Objective.t5] = lambda batch, setting: t5_random_spans_mask_batch(
                batch, **setting, rng=self.rng
            )
            self.mask_generator[Objective.aggressive_t5] = lambda batch, setting: t5_random_spans_mask_batch(
                batch, **setting, rng=self.rng
            )
            self.mask_generator[Objective.prefix] = lambda batch: gpt_span_mask_batch(batch)
            self.mask_generator[Objective.unconditional] = lambda batch: None
        elif self.conditional_generation == "span_infilling":
            self.mask_generator = lambda batch: t5_random_spans_mask_batch(
                batch, data_args.mask_ratio, data_args.mean_mask_span_length, self.rng
            )
        elif self.conditional_generation == "prefix_lm":
            self.mask_generator = lambda batch: gpt_span_mask_batch(
                batch, use_half_length_as_prefix_size=(mode == "eval"), eval_context_size=eval_context_size
            )
        elif self.conditional_generation == "ul2" and mode == "train":
            self.mask_generator = {}
            self.mask_generator[Objective.t5] = lambda batch, setting: t5_random_spans_mask_batch(
                batch, **setting, rng=self.rng
            )
            self.mask_generator[Objective.aggressive_t5] = lambda batch, setting: t5_random_spans_mask_batch(
                batch, **setting, rng=self.rng
            )
            self.mask_generator[Objective.prefix] = lambda batch: gpt_span_mask_batch(batch)
        elif self.conditional_generation == "ul2_variable" and mode == "train":
            self.mask_generator = {}
            self.mask_generator[Objective.t5] = lambda batch, mask_ratio, mean_mask_span_length: t5_random_spans_mask_batch(
                batch, mask_ratio=mask_ratio, mean_mask_span_length=mean_mask_span_length, rng=self.rng
            )
            self.mask_generator[Objective.prefix] = lambda batch: gpt_span_mask_batch(batch)

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
        if self.conditional_generation in ["span_infilling", "prefix_lm"]:
            masks = {"span_mask": self.mask_generator(features)}
        elif self.conditional_generation == "ul2_with_unconditional" and self.mode == "train":
            objectives = [Objective.unconditional, Objective.t5, Objective.prefix, Objective.aggressive_t5]
            weights = [0.25, 0.25, 0.25, 0.25]
            objective = choices(objectives, weights)[0]
            if objective in [Objective.t5, Objective.aggressive_t5]:
                setting = choices(OBJECTIVE_SETTINGS[objective])[0]
                masks = {"span_mask": self.mask_generator[objective](features, setting)}
            else:
                masks = {"span_mask": self.mask_generator[objective](features)}
        elif self.conditional_generation == "ul2" and self.mode == "train":
            objectives = [Objective.t5, Objective.prefix, Objective.aggressive_t5]
            weights = [0.25, 0.25, 0.25]
            objective = choices(objectives, weights)[0]
            if objective in [Objective.t5, Objective.aggressive_t5]:
                setting = choices(OBJECTIVE_SETTINGS[objective])[0]
                masks = {"span_mask": self.mask_generator[objective](features, setting)}
            else:
                masks = {"span_mask": self.mask_generator[objective](features)}
        elif self.conditional_generation == "ul2_variable" and self.mode == "train":
            objectives = [Objective.t5, Objective.prefix]
            weights = [0.5, 0.5]
            objective = choices(objectives, weights)[0]
            if objective  == objective.t5:
                # Here we assume the length is the same for all data in a batch.
                length=len(features[0]["input_ids"])
                min_ratio = 1.0/length            
                mask_ratio = random.uniform(min_ratio, 0.5)
                mean_mask_span_length=int(random.uniform(1, mask_ratio*length))
                masks = {"span_mask": self.mask_generator[objective](features, mask_ratio, mean_mask_span_length)}
            else:
                masks = {"span_mask": self.mask_generator[objective](features)}
        elif self.mode == "eval" and self.conditional_generation in ["ul2", "ul2_with_unconditional", "ul2_variable"]:
            masks = {
                "span_mask": gpt_span_mask_batch(
                    features, use_half_length_as_prefix_size=True, eval_context_size=self.eval_context_size
                )
            }
        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        if "attention_mask" in batch:
            del batch["attention_mask"]
        return {**batch, **masks}


@dataclass
class DataCollatorForSeq2Seq:
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.
    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single sequence
              is provided).
            - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
              acceptable input length for the model if that argument is not provided.
            - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
              lengths).
        max_length (`int`, *optional*):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):
        input_ids = [feature["input_ids"] for feature in features]
        labels = [feature["labels"] for feature in features]
        input_target = [input + target for input, target in zip(input_ids, labels)]
        features = self.tokenizer.pad(
            {"input_ids": input_target},
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        batch_length = features["input_ids"].shape[1]
        masks = [len(input) * [False] + (batch_length - len(input)) * [True] for input in input_ids]
        features["span_mask"] = torch.tensor(masks)
        return features
