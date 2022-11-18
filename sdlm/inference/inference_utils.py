import torch
import torch.nn.functional as F


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
