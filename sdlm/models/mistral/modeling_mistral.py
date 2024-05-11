from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import MaskedLMOutput
from transformers.models.mistral.modeling_mistral import (
    MistralForCausalLM,
    MistralModel,
    MistralPreTrainedModel,
)
from transformers.utils import logging

from sdlm.data.data_collator import DataCollatorForCausalLMSeq2Seq, get_sep_index
from sdlm.data.data_utils import pad_sequence
from sdlm.utils import mix_values_based_on_self_condition

logger = logging.get_logger(__name__)


class Sin(nn.Module):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return torch.sin(input)


class MistralForDiffusionLM(MistralPreTrainedModel):
    _keys_to_ignore_on_save = [r"lm_head.weight", r"lm_head.bias"]
    _keys_to_ignore_on_load_missing = [
        r"lm_head.weight",
        r"lm_head.bias",
    ]
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)
        self.model = MistralModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # self.timestep_embed = nn.Sequential(
        #     nn.Linear(1, config.hidden_size, bias=False),
        #     Sin(),
        #     nn.Linear(config.hidden_size, config.hidden_size, bias=False),
        # )
        self.timestep_embed = nn.Linear(1, config.hidden_size, bias=False)
        self.post_init()

    def post_init(self):
        super().post_init()
        # (un)toggle causal attention
        for decoder_layer in self.model.layers:
            decoder_layer.self_attn.is_causal = self.config.is_causal

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def vocab_to_hidden_dim_embed(self, input_data):
        return F.linear(input_data, self.get_input_embeddings().weight.data.T)

    def forward(
        self,
        timesteps: torch.FloatTensor,
        input_ids: torch.LongTensor,
        simplex: torch.FloatTensor,
        span_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        previous_pred: Optional[torch.FloatTensor] = None,
        reduce_loss: str = "mean",
        **kwargs,
    ) -> Union[Tuple[torch.Tensor], MaskedLMOutput]:
        # simplex -> weighted avg embedding
        inputs_probs = F.softmax(simplex, dim=-1)
        inputs_embeds = self.vocab_to_hidden_dim_embed(inputs_probs)

        if self.config.self_condition is not None:
            if previous_pred is None:
                previous_pred = torch.zeros_like(simplex, device=simplex.device)
            previous_pred_probs = F.softmax(previous_pred, dim=-1)
            if not self.config.self_condition_mix_logits_before_weights:
                previous_pred = self.vocab_to_hidden_dim_embed(previous_pred_probs)
            # In this setting, we mix the probabilities then apply the weight.
            if self.config.self_condition_mix_before_weights:
                mixed_probs = mix_values_based_on_self_condition(
                    self.config.self_condition, inputs_probs, previous_pred_probs
                )
                inputs_embeds = self.vocab_to_hidden_dim_embed(mixed_probs)

        # Original word embeddings without noise.
        inputs_word_embeds = self.get_input_embeddings()(input_ids)
        timesteps = torch.where(span_mask, timesteps, torch.zeros_like(timesteps))
        timesteps_embed = self.timestep_embed(timesteps.unsqueeze(-1).float())
        inputs_embeds = inputs_embeds + timesteps_embed
        # For the unmasked tokens, we only compute their original word embeddings.
        # Note that this also sets the self-conditioned inputs which we are conditioning on
        # to their original word embeddings values.
        inputs_embeds = torch.where(
            span_mask.unsqueeze(-1), inputs_embeds, inputs_word_embeds
        )

        outputs = self.model(
            input_ids=None,  # TODO(rabeeh): we can remove this hack when we moved loss to outside.
            attention_mask=None,  # attention_mask,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=inputs_embeds,
            use_cache=False,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        sequence_output = outputs[0]
        prediction_scores = self.lm_head(sequence_output)
        masked_lm_loss = None

        if input_ids is not None:
            prediction_scores_for_loss = prediction_scores
            loss_fct = CrossEntropyLoss(reduction=reduce_loss)
            labels = (
                torch.where(span_mask, input_ids, -100)
                if span_mask is not None
                else input_ids
            )
            if self.config.mask_padding_in_loss:
                # also mask padding token loss....
                labels = torch.where(labels == self.config.pad_token_id, -100, labels)
            # important: shift labels to the right by one, mimicking the causal pretraining
            labels = labels[:, 1:]
            prediction_scores_for_loss = prediction_scores_for_loss[:, :-1]
            masked_lm_loss = loss_fct(
                prediction_scores_for_loss.reshape(-1, self.config.vocab_size),
                labels.reshape(-1),
            )
            if reduce_loss == "none":
                # take the average loss over tokens, not counting the masked tokens.
                masked_lm_loss = masked_lm_loss.view(input_ids.shape[0], -1)
                masked_lm_loss = masked_lm_loss.sum(dim=-1) / span_mask.sum(dim=-1)

        # shift our logits forward by one, so that input->output match
        prediction_scores = prediction_scores[:, :-1]
        # add back in our start tok.
        padding_pred = torch.zeros_like(prediction_scores[:, 0])[:, None]
        prediction_scores = torch.cat([padding_pred, prediction_scores], dim=1)
        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.last_hidden_state,
            attentions=outputs.attentions,
        )


class MistralForSeq2SeqLM(MistralForCausalLM):
    @torch.inference_mode()
    def generate(self, *args, **kwargs):
        context_tokens = []
        input_ids = kwargs.pop("input_ids")
        SEP = torch.tensor(
            DataCollatorForCausalLMSeq2Seq.MISTRAL_SEP, device=input_ids.device
        )
        for input_id in input_ids:
            # index = list(input_id).index(self.config.eos_token_id)
            end_of_sep_idx = get_sep_index(input_id, SEP)
            context_tokens.append(input_id[: end_of_sep_idx + 1])
        input_ids = pad_sequence(
            context_tokens,
            padding_value=self.config.pad_token_id,
            batch_first=True,
            padding_side=self.config.padding_side,
        )
        kwargs["input_ids"] = input_ids.to(self.device)
        kwargs["attention_mask"] = ~(kwargs["input_ids"] == self.config.pad_token_id)
        kwargs["use_cache"] = False
        outputs = super().generate(*args, **kwargs)
        seq_len = input_ids.size(1)
        output_ids = outputs[:, seq_len:]
        return output_ids.to(self.device)
