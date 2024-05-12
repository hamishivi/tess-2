from typing import Optional

import torch
from torch import autograd
from transformers.modeling_outputs import MaskedLMOutput

from sdlm.models.cdcd.cdf import LossCDF
from sdlm.models.roberta.modeling_roberta import RobertaForDiffusionLM


class CDCDGARRobertaForDiffusionLM(RobertaForDiffusionLM):
    def __init__(self, config):
        super().__init__(config)
        self.cdf = LossCDF(100)

    def apply_gar(
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

    def warp_timesteps(
        self, timesteps: torch.FloatTensor, token_input=None, t_min=0, t_max=1
    ):
        # u has to be in normalized range...
        if t_max - t_min > 0:
            timesteps = (timesteps - t_min) / (t_max - t_min)
        else:
            # weird case, only really happens with 1 diffusion steps (tmin=0,tmax=0)
            # in this case, we just set timesteps to 0
            timesteps = timesteps - t_min
            t_max = 1  # just to avoid div by 0
        # warp timesteps based on gar
        timesteps = self.apply_gar(timesteps, token_input, t_min, t_max)
        # then apply CDF
        return self.cdf(u=timesteps, normalized=True, t_min=t_min, t_max=t_max).detach()

    def forward(
        self,
        timesteps: torch.FloatTensor,
        input_ids: torch.LongTensor,
        simplex: torch.FloatTensor,
        span_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        previous_pred: Optional[torch.FloatTensor] = None,
        classifier_free_guidance: bool = False,
        classifier_free_guidance_in_train: bool = False,
        max_timestep: int = 5000,
        reduce_loss: str = "mean",  # passed to 'reduction' in F.cross_entropy
        # unconditional_simplex: torch.FloatTensor = None,
        return_all_losses: bool = False,  # return per-token loss for all items in batch):
        previous_hidden: Optional[torch.FloatTensor] = None,
        original_timesteps: Optional[torch.FloatTensor] = None,
    ):
        output = super().forward(
            timesteps,
            input_ids,
            simplex,
            span_mask,
            token_type_ids,
            position_ids,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            labels,
            output_attentions,
            output_hidden_states,
            return_dict,
            previous_pred,
            classifier_free_guidance,
            classifier_free_guidance_in_train,
            max_timestep,
            reduce_loss="none",
            return_all_losses=False,
            previous_hidden=previous_hidden,  # for CDCD predictions...
        )
        loss = output.loss
        if self.training:
            # then we learn the cdf from the losses
            # only in train mode, since in eval we just apply the warping.
            new_timesteps_clone = timesteps.clone()
            new_timesteps_clone.requires_grad = True
            with torch.enable_grad():
                # grab the predictions for the loss values - note at this point timesteps
                # are normalised to [0, 1]
                xent_pred = self.cdf(t=new_timesteps_clone, normalized=False, t_max=1)
                # importance weights -> reciprocal of grad of CDF.
                imp_weights = (
                    1.0 / autograd.grad(xent_pred.sum(), [new_timesteps_clone])[0]
                )[:, 0]
            imp_weights = imp_weights.detach() * 1e-5
            # just one index of timesteps since all are the same. required for compat with tokenwise
            cdf_loss = (
                imp_weights
                * (
                    self.cdf(t=timesteps, normalized=False, t_max=1)[:, 0]
                    - loss.detach()
                ).pow(2)
            ).mean()
            loss = loss.mean() + cdf_loss  # upweight cdf loss as its too small :(
        else:
            loss = loss.mean()
        return MaskedLMOutput(
            loss=loss,
            logits=output.logits,
            hidden_states=output.hidden_states,
            attentions=output.attentions,
        )
