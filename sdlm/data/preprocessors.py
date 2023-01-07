"""Implements data preprocessings including the T5 preprocessing."""
import numpy as np
import itertools
import pdb
import torch


# TODO: here the max perhaps needs to be also the half-length.
def gpt_span_mask(length, pad_length, use_half_length_as_prefix_size, eval_context_size):
    """Given the length and pad_length for an input generates a prefix (GPT-style) mask."""
    # Start of the sequence is not masked, so we consider length-1.
    # TODO: we need an assert for length not be smaller than a value.
    if not use_half_length_as_prefix_size:
        # high should be higher than low, otherwise we set prefix_size=1.
        prefix_size = np.random.randint(low=1, high=int((length - 1) / 2)) if length >= 5 else 1
    else:
        # If eval_context_size is set, we consider it, otherwise we use half of the given length as
        # context. Note that since the start token is also masked, we deduct one from the given
        # context size.
        prefix_size = eval_context_size - 1 if eval_context_size is not None else int((length - 1) / 2)
    # The start token is not masked.
    return [False] + [False] * prefix_size + [True] * (length - prefix_size - 1) + [False] * pad_length


def gpt_span_mask_batch(batch, use_half_length_as_prefix_size=False, eval_context_size=None):
    lengths = [len(feature["input_ids"]) for feature in batch]
    max_length = max(lengths)
    masks = [
        gpt_span_mask(length, max_length - length, use_half_length_as_prefix_size, eval_context_size) for length in lengths
    ]
    return torch.tensor(masks)


def t5_random_spans_mask(length, mask_ratio, mean_mask_span_length=3.0, rng=None, pad_length=None):
    """Noise mask consisting of random spans of mask tokens.

    The number of mask tokens and the number of mask spans and non-mask spans
    are determined deterministically as follows:
      num_mask_tokens = round(length * mask_ratio)
      num_nonmask_spans = num_mask_spans = round(
         num_mask_tokens / mean_mask_span_length)
    Spans alternate between non-mask and mask, beginning with non-mask.
    Subject to the above restrictions, all masks are equally likely.
    Note that this function do not mask start/end of sequence.
    Args:
      length: an int32 scalar (length of the incoming token sequence)
      mask_ratio: a float - approximate ratio of output mask (between 0 and 1).
      mean_mask_span_length: Average mask length.
      rng = a np.random.default_rng() instance or None
    Returns:
      a boolean list of shape [length]
    adapted from https://github.com/google-research/text-to-text-transfer-transformer/blob/master/t5/data/preprocessors.py#L2704
    and https://github.com/allenai/contrastive_pretraining/blob/95fe35d3257402c7df362c3e0f746a40d9fba8f0/cpt/data.py#L288
    """
    # By default, we do not maks start and end of sequence.
    # TODO: we need to put assert for this!
    # NOTE: this only works if we use line_by_line which we do not. So I had to remove it.
    # length -= 2
    orig_length = length
    # Increase length to avoid degeneracy.
    length = max(length, 2)

    # Compute number of mask tokens and mask spans.
    num_mask_tokens = int(length * mask_ratio)
    # Avoid degeneracy by ensuring positive numbers of mask and nonmask tokens.
    num_mask_tokens = min(max(num_mask_tokens, 1), length - 1)
    num_mask_spans = int(num_mask_tokens / mean_mask_span_length)
    # Avoid degeneracy by ensuring positive number of mask spans.
    num_mask_spans = max(num_mask_spans, 1)
    num_nonmask_tokens = length - num_mask_tokens
    mask_span_lengths = _random_segmentation(num_mask_tokens, num_mask_spans, rng=rng)
    nonmask_span_lengths = _random_segmentation(num_nonmask_tokens, num_mask_spans, rng=rng)
    mask = list(
        itertools.chain.from_iterable(
            [[False] * nonmask_span_lengths[k] + [True] * mask_span_lengths[k] for k in range(num_mask_spans)]
        )
    )[:orig_length]
    # Start and end of the sequence mask are set to False. Again since this is not line_by_line, we
    # remove this.
    # mask = [False] + mask + [False]
    if pad_length is not None:
        mask += [False for _ in range(pad_length)]
    return mask


def t5_random_spans_mask_batch(batch, mask_ratio, mean_mask_span_length=3.0, rng=None):
    """Given not padded inputs, generates the T5 mask for each input."""
    lengths = [len(feature["input_ids"]) for feature in batch]
    max_length = max(lengths)
    masks = [t5_random_spans_mask(length, mask_ratio, mean_mask_span_length, rng, max_length - length) for length in lengths]
    return torch.tensor(masks)


def _random_segmentation(num_items, num_segments, rng=None):
    """Partition a sequence of items randomly into non-empty segments.
    Args:
      num_items: an integer scalar > 0
      num_segments: an integer scalar in [1, num_items]
      rng = a np.random.default_rng() instance or None
    Returns:
      a list with shape [num_segments] containing positive integers that add up to num_items.
    forked from: https://github.com/allenai/contrastive_pretraining/blob/95fe35d3257402c7df362c3e0f746a40d9fba8f0/cpt/data.py#L265
    """
    first_in_segment = np.arange(num_items - 1) < num_segments - 1
    rng = rng or np.random.default_rng()
    rng.shuffle(first_in_segment)
    # The first position always starts a segment.
    # first_in_segment is boolean array for every position after the first that signals whether this location is the start of a new segment.
    segment_id = np.cumsum(first_in_segment)
    segment_length = [0] * num_segments
    segment_length[0] = 1  # first the first missing first in segment
    for k in range(num_items - 1):
        segment_length[segment_id[k]] += 1
    return segment_length


def insert_extra_paddings(rng, token_ids, pad_token_id, padding_ratio):
    """Inserts padding tokens with the ratio of `padding_ratio` into the token_ids."""
    # TODO: we need to assert to have start/end of sequence tokens.
    # We do not add the padding in the start and end of sequence.
    length = len(token_ids) - 2
    num_padding_tokens = int(length * padding_ratio)
    if num_padding_tokens == 0:
        # In this case, the rate of padding tokens was not enough to add extra tokens.
        return token_ids
    length = length + num_padding_tokens
    # We do not modify the start token.
    all_ids = np.arange(1, length + 1)
    # This is without shuffling.
    # original_ids = np.arange(1, length+1)
    rng = rng or np.random.default_rng()
    rng.shuffle(all_ids)
    # padding tokens positions.
    padding_ids = np.array(all_ids)[:num_padding_tokens] + 1
    token_ids_extended = []
    current_id = 0
    for i in range(length + 2):
        if i not in padding_ids:
            token_ids_extended.append(pad_token_id)
        else:
            token_ids_extended.append(token_ids[current_id])
            current_id += 1
    return token_ids_extended
    """
    # Other tokens positions, we do not change the start and end of sequence tokens.
    other_tokens_ids = [0]+[x for x in original_ids if x not in padding_ids]+[length+1]
    # Considers the start and end of sequence tokens in the final length.
    token_ids_extended = np.full((length+2), pad_token_id, dtype=int)
    token_ids_extended[other_tokens_ids] = token_ids
    return token_ids_extended.tolist()
    """
