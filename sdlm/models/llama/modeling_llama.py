import random
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers.activations import ACT2FN
from transformers.modeling_outputs import MaskedLMOutput
from transformers.models.llama.modeling_llama import (  # RobertaLMHead,
    LlamaModel,
    LlamaPreTrainedModel,
)
from transformers.utils import logging

from sdlm.utils import convert_to_simplex, mix_values_based_on_self_condition

logger = logging.get_logger(__name__)


class LlamaForDiffusionLM(LlamaPreTrainedModel):
    _keys_to_ignore_on_save = [r"lm_head.decoder.weight", r"lm_head.decoder.bias"]
    _keys_to_ignore_on_load_missing = [
        r"position_ids",
        r"lm_head.decoder.weight",
        r"lm_head.decoder.bias",
    ]
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)

        if config.is_decoder:
            logger.warning(
                "If you want to use `RobertaForMaskedLM` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )

        self.model = LlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # # The LM head weights require special treatment only when they are tied with the word embeddings
        # self.update_keys_to_ignore(config, ["lm_head.decoder.weight"])

        self.vocab_to_hidden_dim_embed = nn.Linear(
            config.vocab_size, config.hidden_size, bias=False
        )
        self.timestep_embed = nn.Linear(1, config.hidden_size, bias=True)

        if self.config.self_condition is not None and self.config.deepmind_conditional:
            # In this case, this is self-conditioning with conditional generation as done in DeepMind paper.
            # See Figure 3 in https://arxiv.org/pdf/2211.15089.pdf.
            # Here we concat masked word embeddings, noisy embeddings, mask, and self-conditioning inputs
            # and project them to the hidden_size.
            self.project_to_hidden_size = nn.Linear(
                config.hidden_size * 4, config.hidden_size, bias=False
            )
        elif (
            self.config.self_condition is not None
            and not self.config.self_condition  # noqa: E713
            in [
                "logits_addition",
                "logits_with_projection_addition",
                "logits_max",
                "logits_mean",
            ]
        ):
            if config.self_condition_mlp_projection:
                self.project_to_hidden_size = nn.Sequential(
                    nn.Linear(config.hidden_size * 2, config.hidden_size, bias=False),
                    ACT2FN[config.hidden_act],
                    nn.Linear(config.hidden_size, config.hidden_size, bias=False),
                )
            else:
                self.project_to_hidden_size = nn.Linear(
                    config.hidden_size * 2, config.hidden_size, bias=False
                )

        # Initialize weights and apply final processing
        self.post_init()

    def post_init(self):
        super().post_init()
        self.vocab_to_hidden_dim_embed.weight.data = (
            self.get_input_embeddings().weight.data.T
        )

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

    # TODO: adjust for llama
    # 50264 is hard-coded pad token for roberta
    def get_roberta_empty_tokens(self, shape, device):
        if self.config.empty_token_be_mask:
            empty_token_ids = (
                torch.ones(shape, dtype=torch.int64, device=device) * 50264
            )
        else:
            # Padding token in roberta-large is 1.
            empty_token_ids = torch.ones(shape, dtype=torch.int64, device=device)
        empty_token_ids[:, 0] = 0
        empty_token_ids[:, -1] = 2
        return empty_token_ids

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
        return_all_losses: bool = False,  # return per-token loss for all items in batch
        previous_hidden: Optional[torch.FloatTensor] = None,  # for CDCD predictions...
        original_timesteps: Optional[torch.FloatTensor] = None,
    ) -> Union[Tuple[torch.Tensor], MaskedLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        kwargs (`Dict[str, any]`, optional, defaults to *{}*):
            Used to hide legacy arguments that have been deprecated.
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # If we have a mask, we need to mask the simplex values before softmax.
        """
        if span_mask is not None:
            mask_value = torch.finfo(simplex.dtype).min
            mask_value = torch.tensor(mask_value, dtype=simplex.dtype, device=simplex.device)
            simplex = torch.where(span_mask[:, :, None], simplex, mask_value)
        """
        inputs_probs = F.softmax(simplex, dim=-1)
        inputs_embeds = self.vocab_to_hidden_dim_embed(inputs_probs)

        if classifier_free_guidance or classifier_free_guidance_in_train:
            if self.config.classifier_free_simplex_inputs:
                if self.config.classifier_free_uncond_input == "empty_token":
                    empty_token_ids = self.get_roberta_empty_tokens(
                        shape=input_ids.shape, device=input_ids.device
                    )
                    # TODO: fix the simplex_value later.
                    unconditional_simplex = convert_to_simplex(
                        empty_token_ids, 5.0, self.config.vocab_size
                    )
                elif self.config.classifier_free_uncond_input == "noisy_simplex":
                    simplex_shape = (
                        input_ids.shape[0],
                        input_ids.shape[1],
                        self.config.vocab_size,
                    )
                    unconditional_simplex = 5.0 * torch.randn(
                        simplex_shape, device=input_ids.device
                    )
                else:
                    raise NotImplementedError
                unconditional_probs = F.softmax(unconditional_simplex, dim=-1)
                uncond_inputs_embeds = self.vocab_to_hidden_dim_embed(
                    unconditional_probs
                )
            else:
                empty_token_ids = self.get_roberta_empty_tokens(
                    shape=input_ids.shape, device=input_ids.device
                )
                uncond_inputs_embeds = self.get_input_embeddings()(empty_token_ids)

        if self.config.self_condition is not None:
            if self.config.self_condition_zeros_after_softmax and previous_pred is None:
                previous_pred_probs = torch.zeros_like(simplex, device=simplex.device)
            else:
                if previous_pred is None:
                    previous_pred = torch.zeros_like(simplex, device=simplex.device)
                """
                if span_mask is not None:
                    mask_value = torch.finfo(previous_pred.dtype).min
                    mask_value = torch.tensor(mask_value, dtype=previous_pred.dtype, device=previous_pred.device)
                    previous_pred = torch.where(span_mask[:, :, None], previous_pred, mask_value)
                """
                previous_pred_probs = F.softmax(previous_pred, dim=-1)
            if not self.config.self_condition_mix_logits_before_weights:
                previous_pred = self.vocab_to_hidden_dim_embed(previous_pred_probs)
            if not self.config.deepmind_conditional:
                # In this setting, we mix the probabilities then apply the weight.
                if self.config.self_condition_mix_logits_before_weights:
                    mixed_logits = mix_values_based_on_self_condition(
                        self.config.self_condition, simplex, previous_pred
                    )
                    mixed_probs = F.softmax(mixed_logits, dim=-1)
                    inputs_embeds = self.vocab_to_hidden_dim_embed(mixed_probs)
                elif self.config.self_condition_mix_before_weights:
                    mixed_probs = mix_values_based_on_self_condition(
                        self.config.self_condition, inputs_probs, previous_pred_probs
                    )
                    inputs_embeds = self.vocab_to_hidden_dim_embed(mixed_probs)
                else:
                    if self.config.self_condition in [
                        "logits",
                        "logits_with_projection",
                    ]:
                        inputs_embeds = self.project_to_hidden_size(
                            torch.cat([inputs_embeds, previous_pred], axis=-1)
                        )
                    else:
                        inputs_embeds = mix_values_based_on_self_condition(
                            self.config.self_condition, inputs_embeds, previous_pred
                        )

        if span_mask is not None:
            # Original word embeddings without noise.
            if classifier_free_guidance_in_train and random.uniform(0, 1) < 0.1:
                inputs_word_embeds = uncond_inputs_embeds
            else:
                inputs_word_embeds = self.get_input_embeddings()(input_ids)

        if self.config.self_condition is not None and self.config.deepmind_conditional:
            inputs_embeds = torch.where(
                span_mask.unsqueeze(-1), inputs_embeds, torch.zeros_like(previous_pred)
            )
            previous_pred = torch.where(
                span_mask.unsqueeze(-1), previous_pred, torch.zeros_like(previous_pred)
            )
            inputs_word_embeds = torch.where(
                span_mask.unsqueeze(-1),
                torch.zeros_like(inputs_word_embeds),
                inputs_word_embeds,
            )
            tiled_mask = span_mask.unsqueeze(-1).repeat(1, 1, self.config.hidden_size)
            inputs_embeds = self.project_to_hidden_size(
                torch.cat(
                    [inputs_embeds, inputs_word_embeds, previous_pred, tiled_mask],
                    axis=-1,
                )
            )

        # TODO: remove conversion.
        if len(timesteps.shape) != len(token_rel_positions.shape):
            timesteps = timesteps[:, None]
        # apply token rel pos to transformer timesteps
        timesteps = token_rel_positions * timesteps
        # where we have conditional input we are effectively at the final timesteps
        # we set to 0, since inside everything is scaled to [0,1] (0 = no noise)
        timesteps = torch.where(span_mask, timesteps, torch.zeros_like(timesteps))
        timesteps_embed = self.timestep_embed(timesteps.unsqueeze(-1).float())
        inputs_embeds = inputs_embeds + timesteps_embed

        if span_mask is not None and not self.config.deepmind_conditional:
            # For the unmasked tokens, we only compute their original word embeddings.
            # Note that this also sets the self-conditioned inputs wich we are conditioning on
            # to their original word embeddings values.
            inputs_embeds = torch.where(
                span_mask.unsqueeze(-1), inputs_embeds, inputs_word_embeds
            )
            # TODO: we need to fix classifier-free guidance for the case of deepmind_conditional.
            if classifier_free_guidance:
                inputs_embeds = torch.cat([uncond_inputs_embeds, inputs_embeds])
        outputs = self.model(
            input_ids=None,  # TODO(rabeeh): we can remove this hack when we moved loss to outside.
            attention_mask=None,  # attention_mask,
            # token_type_ids=token_type_ids,
            position_ids=position_ids,
            # head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            # encoder_hidden_states=encoder_hidden_states,
            # encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        prediction_scores = self.lm_head(sequence_output)

        masked_lm_loss = None
        # In case of classifier-free guidance, since the number of output logits and input token ids do not match
        # we do not compute the loss.
        if input_ids is not None:
            # In case of classifier_free guidance we need to get rid of the unconditional part.
            prediction_scores_for_loss = (
                prediction_scores.chunk(2)[1]
                if classifier_free_guidance
                else prediction_scores
            )
            loss_fct = CrossEntropyLoss(reduction=reduce_loss)
            labels = (
                torch.where(span_mask, input_ids, -100)
                if span_mask is not None
                else input_ids
            )
            # also mask padding token loss....
            labels = torch.where(labels == self.config.pad_token_id, -100, labels)
            masked_lm_loss = loss_fct(
                prediction_scores_for_loss.view(-1, self.config.vocab_size),
                labels.view(-1),
            )
            if return_all_losses:
                all_lm_losses = masked_lm_loss.view(input_ids.shape[0], -1)
            if reduce_loss == "none":
                # take the average loss over tokens, not counting the masked tokens.
                masked_lm_loss = masked_lm_loss.view(input_ids.shape[0], -1)
                masked_lm_loss = masked_lm_loss.sum(dim=-1) / span_mask.sum(dim=-1)

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return (
                ((masked_lm_loss,) + output) if masked_lm_loss is not None else output
            )

        return MaskedLMOutput(
            loss=all_lm_losses if return_all_losses else masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.last_hidden_state,
            attentions=outputs.attentions,
        )

    # TODO: adjust for llama rotary
    def resize_position_embeddings(
        self, new_num_position_embeddings: int, with_alternatation=False
    ):
        raise NotImplementedError
