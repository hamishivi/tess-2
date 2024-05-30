import torch.nn as nn
import torch.nn.functional as F
from transformers.models.llama.modeling_llama import (
    LlamaForCausalLM,
    LlamaModel,
    LlamaPreTrainedModel,
)
from transformers.utils import logging

from sdlm.models.mixins.modeling_mixin import (
    CausalLMForSeq2SeqMixin,
    DiffusionModelMixin,
)

logger = logging.get_logger(__name__)


class LlamaForDiffusionLM(DiffusionModelMixin, LlamaPreTrainedModel):
    _keys_to_ignore_on_save = [r"lm_head.weight", r"lm_head.bias"]
    _keys_to_ignore_on_load_missing = [
        r"lm_head.weight",
        r"lm_head.bias",
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


class LlamaForSeq2SeqLM(CausalLMForSeq2SeqMixin, LlamaForCausalLM):
    pass
