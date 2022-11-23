"""Defines the utilities used during the training/infernece of diffusion language models."""
import torch.nn.functional as F
import os
import re
import pdb
from pathlib import Path
from transformers.utils import logging
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
