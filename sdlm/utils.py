"""Defines the utilities used during the training/infernece of diffusion language models."""
import os
import re
import shutil
from pathlib import Path
from typing import Callable, Iterable, List

import numpy as np
import torch
import torch.nn.functional as F
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


def get_last_checkpoint(folder, prefix_checkpoint_dir="step"):
    re_checkpoint = re.compile(r"^" + prefix_checkpoint_dir + r"\_(\d+)$")
    content = os.listdir(folder)
    checkpoints = [
        path
        for path in content
        if re_checkpoint.search(path) is not None
        and os.path.isdir(os.path.join(folder, path))
    ]
    if len(checkpoints) == 0:
        return
    return os.path.join(
        folder, max(checkpoints, key=lambda x: int(re_checkpoint.search(x).groups()[0]))
    )


def remove_checkpoints(output_dir, checkpoint_prefix="step"):
    checkpoints = [
        str(x)
        for x in Path(output_dir).glob(f"{checkpoint_prefix}_*")
        if os.path.isdir(x)
    ]
    for checkpoint in checkpoints:
        logger.info(
            f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit"
        )
        shutil.rmtree(checkpoint)


def get_norm_stats(model):
    # Gradient norm of word embeddings and lm_head.
    input_embed_grad_norm = 0
    if model.roberta.embeddings.word_embeddings.weight.grad is not None:
        input_embed_grad_norm = (
            model.roberta.embeddings.word_embeddings.weight.grad.detach()
            .data.norm(2)
            .item()
        )

    output_embed_grad_norm = 0.0
    if model.lm_head.decoder.weight.grad is not None:
        output_embed_grad_norm = (
            model.lm_head.decoder.weight.grad.detach().data.norm(2).item()
        )

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

# from the open-instruct codebase.
def encode_with_messages_format(
    example, tokenizer, max_seq_length, return_string=False, add_generation_prompt=False
):
    """
    Here we assume each example has a 'messages' field Each message is a dict with 'role' and 'content' fields.
    We concatenate all messages with the roles as delimiters and tokenize them together.
    """
    # we only take the first two messages, since multi-turn is a little more complex
    messages = example["messages"][:2]
    if len(messages) == 0:
        raise ValueError("messages field is empty.")

    def _concat_messages(messages):
        message_text = ""
        for message in messages:
            if message["role"] == "system":
                message_text += "<|system|>\n" + message["content"].strip() + "\n"
            elif message["role"] == "user":
                message_text += "<|user|>\n" + message["content"].strip() + "\n"
            elif message["role"] == "assistant":
                message_text += (
                    "<|assistant|>\n"
                    + message["content"].strip()
                    + tokenizer.eos_token
                    + "\n"
                )
            else:
                raise ValueError("Invalid role: {}".format(message["role"]))
        return message_text

    example_text = tokenizer.bos_token + _concat_messages(messages).strip()
    if add_generation_prompt:
        example_text += '\n<|assistant|>\n'
    tokenized_example = tokenizer(
        example_text,
        add_special_tokens=False,
        return_tensors="pt",
        max_length=max_seq_length,
        truncation=True,
    )
    input_ids = tokenized_example.input_ids
    labels = input_ids.clone()
    if return_string:
        return example_text

    # mask the non-assistant part for avoiding loss
    for message_idx, message in enumerate(messages):
        if message["role"] != "assistant":
            if message_idx == 0:
                message_start_idx = 0
            else:
                message_start_idx = tokenizer(
                    _concat_messages(messages[:message_idx]),
                    return_tensors="pt",
                    max_length=max_seq_length,
                    truncation=True,
                ).input_ids.shape[1]
            if (
                message_idx < len(messages) - 1
                and messages[message_idx + 1]["role"] == "assistant"
            ):
                # here we also ignore the role of the assistant
                messages_so_far = (
                    _concat_messages(messages[: message_idx + 1]) + "<|assistant|>\n"
                )
            else:
                messages_so_far = _concat_messages(messages[: message_idx + 1])
            message_end_idx = tokenizer(
                messages_so_far,
                return_tensors="pt",
                max_length=max_seq_length,
                truncation=True,
                add_special_tokens=False,
            ).input_ids.shape[1]
            # we replace with pad token id,
            labels[:, message_start_idx:message_end_idx] = -100

            if message_end_idx >= max_seq_length:
                break

    attention_mask = torch.ones_like(input_ids)
    return {
        "input_ids": input_ids.flatten(),
        "labels": labels.flatten(),
        "attention_mask": attention_mask.flatten(),
    }
