import torch
from typing import Optional, Tuple, Union
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel, RobertaModel, RobertaLMHead
from transformers.utils import logging
from transformers.modeling_outputs import MaskedLMOutput
from torch.nn import CrossEntropyLoss
import torch.nn as nn
import torch.nn.functional as F
import pdb

logger = logging.get_logger(__name__)


class RobertaForDiffusionLM(RobertaPreTrainedModel):
    _keys_to_ignore_on_save = [r"lm_head.decoder.weight", r"lm_head.decoder.bias"]
    _keys_to_ignore_on_load_missing = [r"position_ids", r"lm_head.decoder.weight", r"lm_head.decoder.bias"]
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)

        if config.is_decoder:
            logger.warning(
                "If you want to use `RobertaForMaskedLM` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.lm_head = RobertaLMHead(config)

        # The LM head weights require special treatment only when they are tied with the word embeddings
        self.update_keys_to_ignore(config, ["lm_head.decoder.weight"])

        self.vocab_to_hidden_dim_embed = nn.Linear(config.vocab_size, config.hidden_size, bias=False)
        self.timestep_embed = nn.Linear(1, config.hidden_size, bias=True)

        if self.config.self_condition is not None:
            self.project_to_half_dimension = nn.Linear(config.hidden_size*2, config.hidden_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def post_init(self):
        super().post_init()
        self.vocab_to_hidden_dim_embed.weight.data = self.get_input_embeddings().weight.data.T

    def get_output_embeddings(self):
        return self.lm_head.decoder

    def set_output_embeddings(self, new_embeddings):
        self.lm_head.decoder = new_embeddings

    def forward(
        self,
        timesteps: torch.FloatTensor,
        input_ids: torch.LongTensor,
        simplex: torch.FloatTensor,
        span_mask: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
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
    ) -> Union[Tuple[torch.Tensor], MaskedLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        kwargs (`Dict[str, any]`, optional, defaults to *{}*):
            Used to hide legacy arguments that have been deprecated.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Adds timesteps to the input simplex.
        inputs_probs = F.softmax(simplex, dim=-1)
        seq_length = inputs_probs.shape[1]
        inputs_embeds = self.vocab_to_hidden_dim_embed(inputs_probs)
        
        if self.config.self_condition in ["logits", "logits_with_projection"]:
            previous_pred_probs = F.softmax(previous_pred, dim=-1)
            previous_pred = self.vocab_to_hidden_dim_embed(previous_pred_probs)
            
        if self.config.self_condition is not None:
            inputs_embeds = self.project_to_half_dimension(torch.cat([inputs_embeds, previous_pred], axis=-1))
        
        # TODO(rabeeh): here this timestep can be improved.
        # TODO: remove conversion.
        timesteps_embed = self.timestep_embed(timesteps.view(-1, 1).float())
        inputs_embeds = inputs_embeds + timesteps_embed.unsqueeze(1).repeat(1, seq_length, 1)

        if span_mask is not None:
            # For the unmasked tokens, we only compute their original word embeddings.
            inputs_word_embeds = self.get_input_embeddings()(input_ids)
            inputs_embeds = torch.where(span_mask.unsqueeze(-1), inputs_embeds, inputs_word_embeds)

        outputs = self.roberta(
            input_ids=None,  # TODO(rabeeh): we can remove this hack when we moved loss to outside.
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        prediction_scores = self.lm_head(sequence_output)

        loss_fct = CrossEntropyLoss()
        masked_lm_loss = None
        if input_ids is not None:
            labels = torch.where(span_mask, input_ids, -100) if span_mask is not None else input_ids
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.last_hidden_state,
            attentions=outputs.attentions,
        )
