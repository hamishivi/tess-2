import torch

from sdlm.models.roberta.modeling_roberta import RobertaForDiffusionLM


# Roberta with the CDF timestep warper.
class GARDiffusionLM(RobertaForDiffusionLM):
    def __init__(self, config):
        super().__init__(config)
        # if true, use my gar warp.
        self.use_gar_warp = True
        self.gar_aggression = 0.5  # [0, 1] range. The higher, the more aggressive the warping (i.e. earlier tokens sent to 0 faster)

    def warp_timesteps(
        self,
        timesteps: torch.FloatTensor,
        token_input=None,
        span_mask=None,
        t_min=0,
        t_max=1,
    ):
        if self.use_gar_warp:
            return self.gar_warp_timesteps(
                timesteps, token_input, span_mask, t_min, t_max
            )
        else:
            return self.ar_warp_timesteps(
                timesteps, token_input, span_mask, t_min, t_max
            )

    def gar_warp_timesteps(
        self,
        timesteps: torch.FloatTensor,
        token_input=None,
        span_mask=None,
        t_min=0,
        t_max=1,
    ):
        # Ensure timesteps is a floating point tensor for computations
        timesteps = timesteps.float()

        # Calculate token masks, excluding specific tokens (masking out padding and special tokens)
        token_masks = ~span_mask

        # Create a tensor representing each position in the sequence [0, 1, ..., seq_len-1]
        seq_len = token_input.size(1)
        positions = torch.arange(seq_len, device=token_input.device).float()

        # Calculate the difference between positions to create a matrix of relative distances
        # Shape of distances: [batch_size, seq_len, seq_len]
        distances = (
            positions.unsqueeze(0).unsqueeze(2) - positions.unsqueeze(0).unsqueeze(1)
        ).abs()
        # Apply token masks to the distances, setting distances for non-masked tokens to 0
        masked_distances = distances * token_masks.unsqueeze(1).float()
        # set non-masked token distances to inf so they don't affect the min operation
        masked_distances = (masked_distances.shape[-1] + 5) * (
            1 - token_masks.unsqueeze(1).float()
        ) + masked_distances

        # Rather than sum, we take the min of the distances (i.e., min distance to a nonmasked token)
        composed = masked_distances.min(dim=2).values
        # set padding tokens to 1, since we dont want these to affect the warping
        # composed = torch.where(
        #     token_input == 1, torch.tensor(1.0, device=token_input.device), composed
        # )
        # normalize to make sure everything is consistent for different lengths.
        composed_max, _ = composed.max(dim=1, keepdim=True)
        composed_normalized = (
            composed / composed_max
        )  # Now composed_normalized is in range [0, 1]
        composed_normalized = (
            1 - composed_normalized
        )  # Invert the composed_normalized values
        composed_normalized = (
            composed_normalized * self.gar_aggression
        )  # Scale the values to range [0, gar_aggression]

        # Adjust timesteps based on composed_normalized values
        # Ensure the operation is broadcastable: [batch_size, 1] * [batch_size, seq_len]
        slope = -t_max / torch.clip(t_max * composed_normalized - t_max, max=1e-8)
        adjusted_timesteps = slope * (timesteps - t_max) + t_max
        adjusted_timesteps = torch.clip(adjusted_timesteps, min=t_min, max=t_max)
        return adjusted_timesteps.long()

    # warp following AR-diffusion paper
    def ar_warp_timesteps(
        self,
        timesteps: torch.FloatTensor,
        token_input=None,
        span_mask=None,
        t_min=0,
        t_max=1,
    ):
        N = 512
        T = t_max
        ne = 2 * N
        te = T
        # Ensure timesteps is a floating point tensor for computations
        timesteps = timesteps.float()
        # rescale timesteps to 0, 1
        timesteps = (timesteps - t_min) / (t_max - t_min)
        # scale up to 0, N+T (for ar-diffusion)
        timesteps = timesteps * (N + T)

        # Create a tensor representing each position in the sequence [0, 1, ..., seq_len-1]
        seq_len = token_input.size(1)
        positions = torch.arange(seq_len, device=token_input.device).float().view(1, -1)
        # based on the span mask, only consider positions we generate for
        input_ends = (1 - span_mask.long()).sum(-1)
        positions = torch.clip(positions - input_ends.view(-1, 1), min=0)

        # calculatute the starting points
        ns = torch.clip(N - timesteps, 0, N)
        ts = torch.clip(timesteps - N, 0, T)
        adjusted_timesteps = torch.clip(
            ((te - ts) / (ne - ns)) * (positions - ns) + ts, 0, T
        )
        # it has been implicitly rescaled to 0, T, so we are done!
        return adjusted_timesteps.long()


# no overriding the forward function, since the warper is deterministic and isn't trained.
