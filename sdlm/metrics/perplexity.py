"""Perplexity Metric. This file is adapted from: https://huggingface.co/spaces/evaluate-measurement/perplexity/blob/main/perplexity.py"""

import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from evaluate import logging
import pdb


def perplexity(
    texts, model, tokenizer, batch_size: int = 16, add_start_token: bool = True, max_length=None, only_return_loss=False
):
    """Perplexity (PPL) can be used for evaluating to what extent a dataset is similar to the distribution of text that
    a given model was trained on. It is defined as the exponentiated average negative log-likelihood of a sequence,
    calculated with exponent base `e`.

    For more information, see https://huggingface.co/docs/transformers/perplexity

    Args:
        texts (list of str): List of text strings.
        model: model used for calculating Perplexity
            NOTE: Perplexity can only be calculated for causal language models.
                    This includes models such as gpt2, causal variations of bert,
                    causal versions of t5, and more (the full list can be found
                    in the AutoModelForCausalLM documentation here:
                    https://huggingface.co/docs/transformers/master/en/model_doc/auto#transformers.AutoModelForCausalLM )
        tokenizer: the corresponding tokenizer for the given model.
        batch_size (int): the batch size to run texts through the model. Defaults to 16.
        add_start_token (bool): whether to add the start token to the texts,
            so the perplexity can include the probability of the first word. Defaults to True.
    Returns:
        perplexity: dictionary containing the perplexity scores for the texts
            in the input list, as well as the mean perplexity. If one of the input texts is
            longer than the max input length of the model, then it is truncated to the
            max length for the perplexity computation.
    """
    device = model.device
    # if batch_size > 1 (which generally leads to padding being required), and
    # if there is not an already assigned pad_token, assign an existing
    # special token to also be the padding token
    if tokenizer.pad_token is None and batch_size > 1:
        existing_special_tokens = list(tokenizer.special_tokens_map_extended.values())
        # check that the model already has at least one special token defined
        assert (
            len(existing_special_tokens) > 0
        ), "If batch_size > 1, model must have at least one special token to use for padding. Please use a different model or set batch_size=1."
        # assign one of the special tokens to also be the pad token
        tokenizer.add_special_tokens({"pad_token": existing_special_tokens[0]})

    if add_start_token and max_length:
        # leave room for <BOS> token to be added:
        assert (
            tokenizer.bos_token is not None
        ), "Input model must already have a BOS token if using add_start_token=True. Please use a different model, or set add_start_token=False"
        max_tokenized_len = max_length - 1
    else:
        max_tokenized_len = max_length

    encodings = tokenizer(
        texts,
        add_special_tokens=False,
        padding=True,
        truncation=True if max_tokenized_len else False,
        max_length=max_tokenized_len,
        return_tensors="pt",
        return_attention_mask=True,
    ).to(device)
    encoded_texts = encodings["input_ids"]
    attn_masks = encodings["attention_mask"]

    # check that each input is long enough:
    if add_start_token:
        assert torch.all(torch.ge(attn_masks.sum(1), 1)), "Each input text must be at least one token long."
    else:
        assert torch.all(
            torch.ge(attn_masks.sum(1), 2)
        ), "When add_start_token=False, each input text must be at least two tokens long. Run with add_start_token=True if inputting strings of only one token, and remove all empty input strings."

    ppls = []
    loss_fct = CrossEntropyLoss(reduction="none")
    if only_return_loss:
        all_losses, all_lengths = [], []
    for start_index in logging.tqdm(range(0, len(encoded_texts), batch_size)):
        end_index = min(start_index + batch_size, len(encoded_texts))
        encoded_batch = encoded_texts[start_index:end_index]
        attn_mask = attn_masks[start_index:end_index]

        if add_start_token:
            bos_tokens_tensor = torch.tensor([[tokenizer.bos_token_id]] * encoded_batch.size(dim=0)).to(device)
            encoded_batch = torch.cat([bos_tokens_tensor, encoded_batch], dim=1)
            attn_mask = torch.cat([torch.ones(bos_tokens_tensor.size(), dtype=torch.int64).to(device), attn_mask], dim=1)

        labels = encoded_batch

        with torch.no_grad():
            out_logits = model(encoded_batch, attention_mask=attn_mask).logits

        shift_logits = out_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_attention_mask_batch = attn_mask[..., 1:].contiguous()

        loss = (loss_fct(shift_logits.transpose(1, 2), shift_labels) * shift_attention_mask_batch).sum(1)
        lengths = shift_attention_mask_batch.sum(1)
        if only_return_loss:
            all_losses.append(loss)
            all_lengths.append(lengths)
        else:
            perplexity_batch = torch.exp(loss / lengths)
            ppls += perplexity_batch.tolist()

    if only_return_loss:
        return all_losses, all_lengths
    else:
        return {"perplexities": ppls, "mean_perplexity": np.mean(ppls)}


def conditional_perplexity(
    texts, prefixes, model, tokenizer, batch_size: int = 16, add_start_token: bool = True, max_length=None
):
    """Computes the conditional perplexity for the case of prefix language modeling."""
    full_texts = [f"{prefix}{text}" for prefix,text in zip(prefixes, texts)]
    loss, lengths = perplexity(full_texts, model, tokenizer, batch_size, add_start_token, max_length, only_return_loss=True)
    prefix_loss, prefix_lengths = perplexity(
        prefixes, model, tokenizer, batch_size, add_start_token, max_length, only_return_loss=True
    )
    # Computing the perplexity over the whole examples.
    ppls = []
    total_nlls = 0
    total_tokens = 0
    for i in range(len(loss)):
        perplexity_batch = torch.exp((loss[i] - prefix_loss[i]) / (lengths[i] - prefix_lengths[i]))
        ppls.extend(perplexity_batch.tolist())
        total_nlls += torch.sum(loss[i] - prefix_loss[i]).item()
        total_tokens += torch.sum(lengths[i] - prefix_lengths[i]).item()
    return {"perplexities": ppls, "mean_perplexity": np.nanmean(ppls), "mean_perplexity_total": np.exp(total_nlls/total_tokens)}
