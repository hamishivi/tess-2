"""Defines the utilities used during the training/infernece of diffusion language models."""
import os
from typing import Callable, Iterable, List

import torch
import torch.nn.functional as F
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import logging

logger = logging.get_logger(__name__)


def join_texts(prefixes, sentences):
    """Joins prefixes to setences."""
    return [f"{prefix}{sentence}" for prefix, sentence in zip(prefixes, sentences)]


def convert_to_simplex(token_ids, simplex_value, vocab_size):
    return 2 * simplex_value * F.one_hot(token_ids, vocab_size) - simplex_value


def scale(inputs, scale_value):
    return inputs / scale_value


def tokenwise_timestep(position, timestep, max_length, max_timesteps):
    n_e, t_e = 2 * max_length, max_timesteps
    n_s = min(max(max_length - timestep, 0), max_length)
    t_s = min(max(timestep - max_length, 0), max_timesteps)
    token_timestep = ((t_e - t_s) / (n_e - n_s)) * (position - n_s) + t_s
    return round(min(max(0, token_timestep), max_timesteps))


def self_condition_preds(self_condition, logits, logits_projection=None):
    if self_condition in [
        "logits",
        "logits_addition",
        "logits_mean",
        "logits_max",
        "logits_multiply",
    ]:
        previous_pred = logits.detach()
    elif self_condition in [
        "logits_with_projection",
        "logits_with_projection_addition",
    ]:
        previous_pred = logits_projection(logits.detach())
    else:
        assert NotImplementedError(f"{self_condition} is not implemented.")
    return previous_pred


def mix_values_based_on_self_condition(self_condition_type, value_1, value_2):
    if self_condition_type in ["logits_with_projection_addition", "logits_addition"]:
        mixed_values = value_1 + value_2
    elif self_condition_type == "logits_mean":
        mixed_values = (value_1 + value_2) / 2.0
    elif self_condition_type == "logits_max":
        mixed_values = torch.max(value_1, value_2)
    elif self_condition_type == "logits_multiply":
        mixed_values = value_1 * value_2
    else:
        assert NotImplementedError
    return mixed_values


def lmap(f: Callable, x: Iterable) -> List:
    """list(map(f, x))"""
    return list(map(f, x))


def pad_data(data_list, tokenizer):
    return tokenizer.pad({"input_ids": data_list}, padding=True)["input_ids"]


def get_last_checkpoint_with_beaker_preemption(training_args) -> str:
    last_checkpoint = None
    if (
        os.path.isdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if (
            last_checkpoint is None
            and len(os.listdir(training_args.output_dir)) > 0
            and not training_args.beaker
        ):
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif (
            last_checkpoint is not None and training_args.resume_from_checkpoint is None
        ):
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )
    return last_checkpoint
