"""Defines the utilities used during the training/infernece of diffusion language models."""
import torch.nn.functional as F
import os
import re
import numpy as np
import pdb
from pathlib import Path
from transformers.utils import logging
from typing import Tuple, Union, Optional
import shutil

logger = logging.get_logger(__name__)


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
    if self_condition in ["logits", "logits_addition"]:
        previous_pred = logits.detach()
    elif self_condition in ["logits_with_projection", "logits_with_projection_addition"]:
        previous_pred = logits_projection(logits.detach())
    else:
        assert NotImplementedError(f"{self_condition} is not implemented.")
    return previous_pred


class EvalPrediction:
    """
    Evaluation output to be used to compute metrics.
    Parameters:
        logits (`np.ndarray`): Logits predictions of the model.
        simplex (`np.ndarray`): Simplex predictions of the model.
        inputs (`np.ndarray`, *optional*)
    """

    def __init__(
        self,
        logits: Union[np.ndarray, Tuple[np.ndarray]],
        simplex: Union[np.ndarray, Tuple[np.ndarray]],
        inputs: Optional[Union[np.ndarray, Tuple[np.ndarray]]] = None,
    ):
        self.logits = logits
        self.simplex = simplex
        self.inputs = inputs

    def __iter__(self):
        if self.inputs is not None:
            return iter((self.logits, self.simplex, self.inputs))
        else:
            return iter((self.logits, self.simplex))

    def __getitem__(self, idx):
        if idx < 0 or idx > 2:
            raise IndexError("tuple index out of range")
        if idx == 2 and self.inputs is None:
            raise IndexError("tuple index out of range")
        if idx == 0:
            return self.simplex
        elif idx == 1:
            return self.logits
        elif idx == 2:
            return self.inputs
