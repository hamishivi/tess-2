"""Defines the utilities used during the training/infernece of diffusion language models."""
import torch.nn.functional as F
import os
import re
import pdb
from pathlib import Path
from transformers.utils import logging
import shutil
import numpy as np
from typing import Callable, Iterable, List
import torch

logger = logging.get_logger(__name__)


def join_texts(prefixes, sentences):
    """Joins prefixes to setences."""
    return [f"{prefix}{sentence}" for prefix, sentence in zip(prefixes, sentences)]


def convert_to_simplex(token_ids, simplex_value, vocab_size):
    return 2 * simplex_value * F.one_hot(token_ids, vocab_size) - simplex_value


def scale(inputs, scale_value):
    return inputs / scale_value


def get_last_checkpoint(folder, prefix_checkpoint_dir="step"):
    re_checkpoint = re.compile(r"^" + prefix_checkpoint_dir + r"\_(\d+)$")
    content = os.listdir(folder)
    checkpoints = [
        path for path in content if re_checkpoint.search(path) is not None and os.path.isdir(os.path.join(folder, path))
    ]
    if len(checkpoints) == 0:
        return
    return os.path.join(folder, max(checkpoints, key=lambda x: int(re_checkpoint.search(x).groups()[0])))


def remove_checkpoints(output_dir, checkpoint_prefix="step"):
    checkpoints = [str(x) for x in Path(output_dir).glob(f"{checkpoint_prefix}_*") if os.path.isdir(x)]
    for checkpoint in checkpoints:
        logger.info(f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit")
        shutil.rmtree(checkpoint)


def get_norm_stats(model):
    # Gradient norm of word embeddings and lm_head.
    input_embed_grad_norm = 0
    if model.roberta.embeddings.word_embeddings.weight.grad is not None:
        input_embed_grad_norm = model.roberta.embeddings.word_embeddings.weight.grad.detach().data.norm(2).item()

    output_embed_grad_norm = 0.0
    if model.lm_head.decoder.weight.grad is not None:
        output_embed_grad_norm = model.lm_head.decoder.weight.grad.detach().data.norm(2).item()

    """
    total_grad_norm = 0.0
    for p in model.parameters():
        grad_norm = 0.0
        if  p.grad is not None:
            grad_norm = p.grad.detach().data.norm(2).item()
        total_grad_norm += grad_norm ** 2
    total_grad_norm = total_grad_norm ** 0.5

    # Norms of word embeddings and lm_head.
    input_embed_norm = model.roberta.embeddings.word_embeddings.weight.detach().data.norm(2).item()
    output_embed_norm = model.lm_head.decoder.weight.detach().data.norm(2).item()
    total_param_norm = 0.0
    for p in model.parameters():
        param_norm = p.detach().data.norm(2)
        total_param_norm += param_norm.item() ** 2
    total_param_norm = total_param_norm ** 0.5
    """
    return {
        "input_embed_grad_norm": input_embed_grad_norm,
        "output_embed_grad_norm": output_embed_grad_norm,
        # "total_grad_norm": total_grad_norm,
        # "input_embed_norm": input_embed_norm,
        # "output_embed_norm": output_embed_norm,
        # "total_param_norm": total_param_norm
    }


def self_condition_preds(self_condition, logits, logits_projection=None):
    if self_condition in ["logits", "logits_addition", "logits_mean", "logits_max", "logits_multiply"]:
        previous_pred = logits.detach()
    elif self_condition in ["logits_with_projection", "logits_with_projection_addition"]:
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

def round_stsb_target(label):
    """STSB maps two sentences to a floating point number between 1 and 5
    representing their semantic similarity. Since we are treating all tasks as
    text-to-text tasks we need to convert this floating point number to a string.
    The vast majority of the similarity score labels in STSB are in the set
    [0, 0.2, 0.4, ..., 4.8, 5.0]. So, we first round the number to the closest
    entry in this set, and then we convert the result to a string (literally e.g.
    "3.4"). This converts STSB roughly into a 26-class classification dataset.
    Args:
      label: original label.
    Returns:
      A preprocessed label.
    """
    return np.round((label * 5) / 5, decimals=1)


def lmap(f: Callable, x: Iterable) -> List:
    """list(map(f, x))"""
    return list(map(f, x))


def pad_data(data_list, tokenizer):
    return tokenizer.pad({"input_ids": data_list}, padding=True)["input_ids"]
