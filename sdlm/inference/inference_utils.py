import torch
import torch.nn.functional as F
import numpy as np
import pdb
from sdlm.utils import convert_to_simplex
from sdlm.metrics.perplexity import perplexity, conditional_perplexity
from sdlm.metrics.metrics import distinct_n_grams, mauve


def sample_logits(sampling_type, logits, top_p):
    # top-p (nucleus) sampling.
    if sampling_type == "top_p":
        probs = F.softmax(logits, dim=-1)
        sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
        cumsum_probs = torch.cumsum(sorted_probs, dim=-1)

        # Remove tokens with cumulative probability above the threshold.
        sorted_indices_to_keep = cumsum_probs < top_p

        # Shift the indices to the right to keep also the first token below the threshold.
        sorted_indices_to_keep[..., 1:] = sorted_indices_to_keep[..., :-1].clone()
        sorted_indices_to_keep[..., 0] = 1

        indices_to_keep = sorted_indices_to_keep.scatter(dim=2, index=sorted_indices, src=sorted_indices_to_keep)
        filtered_logits = logits.masked_fill(indices_to_keep == 0, -float("Inf"))

        # sample from the filtered distribution.
        token_ids = torch.distributions.categorical.Categorical(logits=filtered_logits).sample()
    else:
        assert NotImplementedError
    return token_ids


def remove_first_occurrence(string, char):
    if char in string:
        idx = string.index(char)
        string = string[idx + len(char) :]
    return string.strip()


def keep_till_first_occurrence(string, chars):
    """Given a list of characters, trim the text after the first occurance between them."""
    idxs = [string.index(char) for char in chars if char in string]
    if len(idxs):
        min_idx = np.min(idxs)
        string = string[:min_idx]
    return string.strip()


def process_text(texts):
    # TODO(rabeeh): for now we only cover roberta case.
    texts = [keep_till_first_occurrence(text, ["</s>"]) for text in texts]
    texts = [remove_first_occurrence(text, "<s>") for text in texts]
    return texts


def split_into_masked_and_unmasked(token_ids, span_mask, return_masked=None):
    """Given an span_mask, splits the given token_ids into masked and unmasked parts.

    If return_masked is set, only returns the masked parts, if this is set to False,
    only returns the unmasked parts, and If set to None, returns both parts.
    """

    def update_spans(span, masked, unmasked, mask):
        # TODO: this needs to be here for previous version of the codes.
        # span = torch.stack(span)
        masked.append(span) if mask else unmasked.append(span)

    masked = []
    unmasked = []
    prev_mask = span_mask[0]
    span = []
    for _, (token_id, mask) in enumerate(zip(token_ids, span_mask)):
        if mask == prev_mask:
            span.append(token_id)
        else:
            # Adds the previous span.
            update_spans(span, masked, unmasked, prev_mask)
            prev_mask = mask
            span = [token_id]
    # Adds the last span.
    update_spans(span, masked, unmasked, prev_mask)

    if return_masked is None:
        return masked, unmasked

    return masked if return_masked else unmasked


def concatenate_alternatively(longer, shorter, mark=""):
    """Given two lists of strings, concatenates them alternatively.

    We assume that the concatenated string should starts from elements in the longer
    list (which has one extra element). The shorter text can optionally be embraced with
    a `mark` text on both sides.
    """
    concatenated_str = ""
    for l, s in zip(longer, shorter):
        concatenated_str += l + " " + mark + s + mark + " "
    if len(longer) == len(shorter) + 1:
        return concatenated_str + longer[-1]
    elif len(longer) == len(shorter):
        return concatenated_str[:-1]
    else:
        raise ValueError


def logits_projection(logits, sampling_type, top_p, simplex_value):
    # TODO(rabeeh): huggingface has different sampling, like constrastive one.
    # also there are more variant in diffusion-lm.
    token_ids = sample_logits(sampling_type, logits, top_p)
    return convert_to_simplex(token_ids, simplex_value, vocab_size=logits.shape[2])


def filter_empty(texts):
    """Filters empty texts and return the remained texts and the their indices."""
    list_of_tuples = [(text, i) for i, text in enumerate(texts) if text != ""]
    texts, remained_inds = list(zip(*list_of_tuples))
    return list(texts), list(remained_inds)


def predict_conditional_generated(span_masks, input_ids, tokenizer, predicted_token_ids, prefix_name):
    masked = list(
        map(lambda x, y: split_into_masked_and_unmasked(x, y, return_masked=True), predicted_token_ids, span_masks)
    )
    unmasked = list(map(lambda x, y: split_into_masked_and_unmasked(x, y, return_masked=False), input_ids, span_masks))
    pred_masked_texts = [tokenizer.batch_decode(x, skip_special_tokens=False) for x in masked]
    pred_unmasked_texts = [tokenizer.batch_decode(x, skip_special_tokens=False) for x in unmasked]
    pred_texts = list(map(lambda x, y: concatenate_alternatively(x, y), pred_unmasked_texts, pred_masked_texts))
    pred_texts_marked = list(
        map(lambda x, y: concatenate_alternatively(x, y, mark="***"), pred_unmasked_texts, pred_masked_texts)
    )
    return {prefix_name: pred_texts, prefix_name + "_marked": pred_texts_marked}


def evaluate_generation(
    results,
    causal_model,
    causal_tokenizer,
    is_conditional_generation,
    prefix_lm=False,
):
    metrics = {}
    # In case of evaluating the results of gpt2, then we only have the "gpt2_texts".
    keys = ["gpt2_texts"] if "gpt2_texts" in results else ["pred_texts_from_simplex", "pred_texts_from_logits"]
    if prefix_lm:
        prefixes = results["prefixes"]
    if is_conditional_generation:
        gold_texts = process_text(results["gold_texts"])
    for key in keys:
        key_metrics = {}
        texts = results[key]
        texts = process_text(texts)
        texts, remained_indices = filter_empty(texts)
        if len(texts) == 0:
            continue

        # Perplexity measured by a causal model. In case of prefix_lm, we compute the conditional perplexity.
        if prefix_lm:
            key_metrics.update(
                {"perplexity": conditional_perplexity(texts, prefixes, causal_model, causal_tokenizer)["mean_perplexity"]}
            )
        else:
            key_metrics.update({"perplexity": perplexity(texts, causal_model, causal_tokenizer)["mean_perplexity"]})
        # Dist-1,2,3 measurements.
        key_metrics.update(distinct_n_grams(texts))
        # Metrics requiring the gold text.
        if is_conditional_generation:
            # Note that we need to pass both context and predicted texts to this metric.
            remained_gold_texts = [text for i, text in enumerate(gold_texts) if i in remained_indices]
            key_metrics.update(mauve(predictions=texts, references=remained_gold_texts))

        # Adds the metrics.
        key_metrics = {f"{key}_{k}": v for k, v in key_metrics.items()}
        metrics.update(key_metrics)

    return metrics
