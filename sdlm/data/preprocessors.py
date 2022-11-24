"""Implements data preprocessings including the T5 preprocessing."""
import numpy as np
import itertools


def t5_random_spans_mask(length, mask_ratio, mean_mask_span_length=3.0, rng=None):
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
    length -= 2
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
    # Start and end of the sequence mask are set to Faslse.
    return [False] + mask + [False]

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
