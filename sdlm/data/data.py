"""Implements different dataset transformation like span-infilling for pre-training diffusion models."""
import torch
import numpy as np
from sdlm.data.preprocessors import t5_random_spans_mask

class SpanInfillingDataset(torch.utils.data.Dataset):
    """Span denoising dataset."""

    def __init__(self, dataset, max_seq_length, mask_ratio=0.15, mean_mask_span_length=3.0, seed=42):
        self.dataset = dataset
        self.max_seq_length = max_seq_length
        self.mask_ratio = mask_ratio
        self.mean_mask_span_length = mean_mask_span_length
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int):
        sample = self.dataset[idx]
        # TODO: maybe this should be actual seq length!
        # Generate a mask for span infilling. Note that we do not mask the start and the end of
        # sequence tokens. Also note that this function should be called after adding the special
        # tokens to the sequence.
        mask = t5_random_spans_mask(
            self.max_seq_length, self.mask_ratio, self.mean_mask_span_length, self.rng
        )
        yield {**sample, "span_mask": torch.tensor(mask)}