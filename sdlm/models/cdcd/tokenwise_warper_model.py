from typing import Optional

import numpy as np
import torch
from torch import autograd
from transformers import RobertaForMaskedLM
from transformers.modeling_outputs import MaskedLMOutput

from sdlm.models.cdcd.cdf import LossCDF
from sdlm.models.roberta.configuration_roberta import RobertaDiffusionConfig
from sdlm.models.roberta.modeling_roberta import RobertaForDiffusionLM


# only difference is that we add n_bins to the config
class TokenwiseCDCDRobertaConfig(RobertaDiffusionConfig):
    def __init__(self, *args, n_bins=100, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_bins = n_bins


# Roberta with the CDF timestep warper.
class TokenwiseCDCDRobertaForDiffusionLM(RobertaForDiffusionLM):
    def __init__(self, config):
        super().__init__(config)
        self.cdf = LossCDF(100)
        # keep the hidden dim larger?
        self.base_lm = RobertaForMaskedLM.from_pretrained("roberta-base")
        self.linear_lu = torch.nn.Sequential(
            torch.nn.Linear(self.config.hidden_size, self.config.hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(self.config.hidden_size, 100),
        )
        self.linear_lt = torch.nn.Sequential(
            torch.nn.Linear(self.config.hidden_size, self.config.hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(self.config.hidden_size, 100),
        )
        self.start_lt = torch.zeros([100]) - float(np.log(100))
        self.start_lu = torch.zeros([100]) - float(np.log(100))
        # small starting a
        self.linear_lu_start_a = torch.nn.Parameter(torch.zeros([1]) + 1)
        self.linear_lt_start_a = torch.nn.Parameter(torch.zeros([1]) + 1)

    def warp_timesteps(
        self,
        timesteps: torch.FloatTensor,
        token_input: Optional[torch.LongTensor] = None,
        t_min=0,
        t_max=1,
    ):
        # u has to be in normalized range...
        if t_max - t_min > 0:
            timesteps = (timesteps - t_min) / (t_max - t_min)
        else:
            # weird case, only really happens with 1 diffusion steps (tmin=0,tmax=0)
            # in this case, we just set timesteps to 0
            timesteps = timesteps - t_min
            t_max = 1  # just to avoid div by 0
        if token_input is None:
            lu, lt = None, None
        else:
            # replace padding tokens with <mask> token
            # to avoid model ignoring those tokens
            token_input = torch.where(token_input == 1, 50264, token_input)
            hidden_states = self.base_lm.roberta(
                input_ids=token_input, output_hidden_states=True
            ).hidden_states[-1]
            # predict out the new timesteps
            lu = self.start_lu.to(
                self.linear_lu_start_a.device
            ) + self.linear_lu_start_a * self.linear_lu(
                torch.cat([hidden_states], dim=-1)
            )
            lt = self.start_lt.to(
                self.linear_lu_start_a.device
            ) + self.linear_lt_start_a * self.linear_lt(
                torch.cat([hidden_states], dim=-1)
            )
            # lu = self.linear_lu(previous_hidden)
            # lt = self.linear_lt(previous_hidden)
        # warp timesteps. sep. call so we can pass to scheduler
        # detach so we don't backprop through this
        return self.cdf(
            u=timesteps, normalized=True, t_min=t_min, t_max=t_max, l_u=lu, l_t=lt
        ).detach()

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
        token_rel_positions: Optional[torch.LongTensor] = None,
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
            token_rel_positions,
            classifier_free_guidance_in_train,
            max_timestep,
            reduce_loss="none",
            return_all_losses=True,
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
                # at train time: we want to predict the tokens not in the span mask,
                # replace with <mask>
                token_input = torch.where((input_ids * span_mask) > 1, 50264, input_ids)
                previous_hidden = self.base_lm.roberta(
                    input_ids=token_input, output_hidden_states=True
                ).hidden_states[-1]
                if previous_hidden is None:
                    lu, lt = None, None
                else:
                    lu = self.start_lu.to(
                        self.linear_lu_start_a.device
                    ) + self.linear_lu_start_a * self.linear_lu(
                        torch.cat([previous_hidden], dim=-1)
                    )
                    lt = self.start_lt.to(
                        self.linear_lt_start_a.device
                    ) + self.linear_lt_start_a * self.linear_lt(
                        torch.cat([previous_hidden], dim=-1)
                    )
                    # lu = self.linear_lu(previous_hidden)
                    # lt = self.linear_lt(previous_hidden)
                xent_pred = self.cdf(
                    t=new_timesteps_clone, normalized=False, t_max=1, l_u=lu, l_t=lt
                )
                # importance weights -> reciprocal of grad of CDF.
                imp_weights = (
                    1.0 / autograd.grad(xent_pred.sum(), [new_timesteps_clone])[0]
                )
            imp_weights = imp_weights.detach() * 1e-5
            cdf_loss = imp_weights * (
                self.cdf(t=timesteps, normalized=False, t_max=1, l_u=lu, l_t=lt)
                - loss.detach()
            ).pow(2)
            # mask regular input part of loss, since we don't warp this anyway.
            # also mask out padding at the end.
            cdf_loss = cdf_loss * span_mask * (input_ids != 1)
            import pdb

            pdb.set_trace()
            loss = loss.mean() + cdf_loss.mean()
        else:
            loss = loss.mean()
        return MaskedLMOutput(
            loss=loss,
            logits=output.logits,
            hidden_states=output.hidden_states,
            attentions=output.attentions,
        )
