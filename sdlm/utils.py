"""Defines the utilities used during the training/infernece of diffusion language models."""
import os
from typing import Callable, Iterable, List
from collections import defaultdict

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

# from the open-instruct codebase.
# NOTE: this is only used for eval and ar training
def encode_with_messages_format_v1(
    example, tokenizer, max_seq_length, return_string=False, add_generation_prompt=False
):
    """
    Here we assume each example has a 'messages' field Each message is a dict with 'role' and 'content' fields.
    We concatenate all messages with the roles as delimiters and tokenize them together.
    """
    # filter (open orca)
    messages = [
        message
        for message in example["messages"]
        if message["role"] in {"user", "assistant"}
    ]
    # we only take the first two messages, since multi-turn is a little more complex
    messages = messages[:2]

    if len(messages) == 0:
        raise ValueError("messages field is empty.")
    # quick sanity checks
    assert messages[0]["role"] == "user"

    def _concat_messages(messages):
        message_text = ""
        for message in messages:
            if message["role"] == "user":
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
        example_text += "\n<|assistant|>\n"
    if return_string:
        return example_text
    tokenized_example = tokenizer(
        example_text,
        add_special_tokens=False,
        return_tensors="pt",
        max_length=max_seq_length,
        truncation=True,
    )
    input_ids = tokenized_example.input_ids
    labels = input_ids.clone()

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


# fixes some newline issues in v1
# NOTE: this is only used for training
def encode_with_messages_format_v2(
    messages,
    tokenizer,
    max_seq_length: int,
):
    """
    `encode_with_messages_format`, but with prefix-accumulating multiturn format
    ex) input_ids: (a1, b1, a2, b2, a3), labels: (b3)
    """
    # quick sanity checks
    if len(messages) == 0:
        raise ValueError("messages field is empty.")
    assert messages[0]["role"] == "user"
    assert messages[1]["role"] == "assistant"

    # double check tokenizer config
    assert tokenizer.add_bos_token
    assert not tokenizer.add_eos_token
    assert tokenizer.padding_side == "right"

    message_text = tokenizer.bos_token
    result = defaultdict(list)
    for message in messages:
        if message["role"] == "user":
            message_text += "<|user|>\n" + message["content"].strip() + "\n"
        elif message["role"] == "assistant":
            # tokenize message so far as context
            # add generation prompt to mask out from loss
            tokenized_context = tokenizer(
                message_text + "<|assistant|>\n",
                truncation=False,
                padding=False,
                add_special_tokens=False,
            )
            context_length = len(tokenized_context["input_ids"])

            if context_length >= max_seq_length:
                break

            # append label
            message_text += "<|assistant|>\n" + message["content"].strip()

            # tokenize full message text
            # add eos and pad
            tokenized_example = tokenizer(
                (message_text + tokenizer.eos_token).strip(),
                truncation=True,
                padding="max_length",
                max_length=max_seq_length,
                return_tensors="pt",
                add_special_tokens=False,
            )
            input_ids = tokenized_example["input_ids"].squeeze()
            labels = input_ids.clone()
            labels[:context_length] = -100
            result["input_ids"].append(input_ids)
            result["labels"].append(labels)

            # add newline for next turn
            message_text += "\n"

    if not result:
        return result
    result["input_ids"] = torch.stack(result["input_ids"])
    result["labels"] = torch.stack(result["labels"])
    return result


# batched version of encode_with_messages_format_v2
def encode_with_messages_format_v2_batch(
    batch,
    tokenizer,
    max_seq_length: int,
    is_tulu_pair: bool = False,
    is_tulu_multiturn: bool = False,
    is_tulu_sliding_window_multiturn: bool = False,
):
    result = {"input_ids": [], "labels": []}

    def _helper(messages):
        encoded = encode_with_messages_format_v2(
            messages=messages,
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
        )
        for key, value in encoded.items():
            result[key].append(value)

    for messages in batch["messages"]:
        # filter (open orca)
        messages = [
            message for message in messages if message["role"] in {"user", "assistant"}
        ]
        if is_tulu_multiturn:
            _helper(messages)
        elif is_tulu_sliding_window_multiturn:
            for i in range(0, len(messages) - 1, 2):
                _helper(messages[i:])
        else:
            max_message_idx = len(messages) - 1 if is_tulu_pair else 2
            for i in range(0, max_message_idx, 2):
                _helper(messages[i : i + 2])
    if result["input_ids"]:
        result["input_ids"] = torch.cat(result["input_ids"], dim=0)
        result["labels"] = torch.cat(result["labels"], dim=0)
    return result



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


def resolve_last_checkpoint_vs_resume_from_checkpoint(
    last_checkpoint, resume_from_checkpoint
):
    """
    Prioritizes last_checkpoint over resume_from_checkpoint.
    When a job configured with `resume_from_checkpoint` is preempted and restarted,
    it needs to start from the last checkpoint in the beaker dataset, not the checkpoint
    specified via `resume_from_checkpoint`; otherwise we lose all progress made in the previous job.
    """
    checkpoint = None
    if last_checkpoint is not None:
        checkpoint = last_checkpoint
    elif resume_from_checkpoint is not None:
        checkpoint = resume_from_checkpoint
    return checkpoint


def is_weka_available() -> bool:
    # assume mount path is /data/input
    # jupiter
    return os.path.isdir("/data/input")


def is_nfs_available() -> bool:
    # allennlp, a100, pluto
    return os.path.isdir("/net/nfs.cirrascale")


def set_hf_home() -> None:
    if is_weka_available():
        os.environ["HF_HOME"] = "/data/input/jaket/.hf"
    elif is_nfs_available():
        os.environ["HF_HOME"] = "/net/nfs.cirrascale/allennlp/jaket/.hf"


def set_pretraining_dataset(data_args) -> None:
    if is_weka_available():
        data_args.dataset_name = "sdlm/data/dolma/dolma_dataset.py"
    else:
        data_args.dataset_name = "emozilla/dolma-v1_7-305B"
