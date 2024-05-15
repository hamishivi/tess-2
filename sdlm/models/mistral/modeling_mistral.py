import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.mistral.modeling_mistral import (
    MistralForCausalLM,
    MistralModel,
    MistralPreTrainedModel,
)
from transformers.utils import logging

from sdlm.data.data_collator import DataCollatorForCausalLMSeq2Seq, get_sep_index
from sdlm.data.data_utils import pad_sequence
from sdlm.models.mixins.modeling_mixin import DiffusionModelMixin

logger = logging.get_logger(__name__)


class Sin(nn.Module):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return torch.sin(input)


class MistralForDiffusionLM(DiffusionModelMixin, MistralPreTrainedModel):
    _keys_to_ignore_on_save = [r"lm_head.weight", r"lm_head.bias"]
    _keys_to_ignore_on_load_mi2ssing = [
        r"lm_head.weight",
        r"lm_head.bias",
    ]
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)
        self.model = MistralModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if not self.config.disable_timestep_embed:
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
