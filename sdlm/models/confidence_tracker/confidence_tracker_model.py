from typing import Optional

import torch
from transformers.modeling_outputs import MaskedLMOutput

from sdlm.models.roberta.modeling_roberta import RobertaForDiffusionLM


# Roberta with the confidence tracker. empirically the same,
# but alters timesteps based on last confidence.
# operates on a token level.
class ConfidenceTrackerRobertaDiffusionLM(RobertaForDiffusionLM):
    def __init__(self, config):
        super().__init__(config)

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
        last_confidence_scores: Optional[torch.FloatTensor] = None,
    ):
        # main difference: timesteps are the min(1-confidence, timesteps)
        # 1 - since 1 is full noise.
        # if last_confidence_scores is not None:
        #     timesteps = torch.min(
        #         torch.where(last_confidence_scores > 0.99, 1 - last_confidence_scores, timesteps), timesteps
        #     )
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
            reduce_loss=reduce_loss,
            return_all_losses=False,
        )
        loss = output.loss.mean()
        # confidence = how much did we put on the right token?
        # todo: calibrate this to the right scale.
        confidence_scores = torch.softmax(output.logits, dim=-1).max(dim=-1).values
        if not self.training:
            return (
                MaskedLMOutput(
                    loss=loss,
                    logits=output.logits,
                    hidden_states=output.hidden_states,
                    attentions=output.attentions,
                ),
                confidence_scores,
            )
        else:
            return MaskedLMOutput(
                loss=loss,
                logits=output.logits,
                hidden_states=output.hidden_states,
                attentions=output.attentions,
            )
