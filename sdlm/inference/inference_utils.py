import torch
import torch.nn.functional as F
import numpy as np
import pdb
from sdlm.utils import convert_to_simplex

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
        span = torch.stack(span)
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
    list which has one extra element. The shorter text can optionally be embraced with
    a `mark` text on both sides.
    """
    assert len(longer) == len(shorter) + 1
    concatenated_str = ""
    for l, s in zip(longer, shorter):
        concatenated_str += l + " " + mark + s + mark + " "
    return concatenated_str + longer[-1]


def logits_projection(logits, sampling_type, top_p, simplex_value):
    # TODO(rabeeh): huggingface has different sampling, like constrastive one.
    # also there are more variant in diffusion-lm.
    token_ids = sample_logits(sampling_type, logits, top_p)
    return convert_to_simplex(token_ids, simplex_value, vocab_size=logits.shape[2])
