import torch

from sdlm.models.roberta.modeling_roberta import RobertaForDiffusionLM


# Roberta with the CDF timestep warper.
class GARDiffusionLM(RobertaForDiffusionLM):
    def __init__(self, config):
        super().__init__(config)

    def warp_timesteps(
        self, timesteps: torch.FloatTensor, token_input=None, t_min=0, t_max=1
    ):
        # Ensure timesteps is a floating point tensor for computations
        timesteps = timesteps.float()

        # Calculate token masks, excluding specific tokens (masking out padding and special tokens)
        token_masks = (token_input != 50264) & (token_input != 1)

        # Create a tensor representing each position in the sequence [0, 1, ..., seq_len-1]
        seq_len = token_input.size(1)
        positions = torch.arange(seq_len, device=token_input.device).float()

        # Calculate the difference between positions to create a matrix of relative distances
        # Shape of distances: [batch_size, seq_len, seq_len]
        distances = positions.unsqueeze(0).unsqueeze(2) - positions.unsqueeze(
            0
        ).unsqueeze(1)
        distances = distances.abs() / (
            seq_len - 1
        )  # Normalize distances to range [0, 1]

        # Apply token masks to the distances, setting distances for masked tokens to 0
        masked_distances = distances * token_masks.unsqueeze(1).float()

        # Sum the distances for each position, then normalize by the maximum distance to ensure range [0, 1]
        composed = masked_distances.sum(dim=2)
        # set padding tokens to 1, since we dont want these to affect the warping
        composed = torch.where(
            token_input == 1, torch.tensor(1.0, device=token_input.device), composed
        )
        composed_max, _ = composed.max(dim=1, keepdim=True)
        composed_normalized = (
            composed / composed_max
        )  # Now composed_normalized is in range [0, 1]
        composed_normalized = (
            1 - composed_normalized
        )  # Invert the composed_normalized values
        composed_normalized = (
            composed_normalized * 0.5
        )  # Scale the values to range [0, 0.5]

        # Adjust timesteps based on composed_normalized values
        # Ensure the operation is broadcastable: [batch_size, 1] * [batch_size, seq_len]
        slope = -t_max / torch.clip(t_max * composed_normalized - t_max, max=1e-8)
        adjusted_timesteps = slope * (timesteps - t_max) + t_max
        adjusted_timesteps = torch.clip(adjusted_timesteps, min=t_min, max=t_max)
        return adjusted_timesteps.long()


# no overriding the forward function, since the warper is deterministic and isn't trained.
