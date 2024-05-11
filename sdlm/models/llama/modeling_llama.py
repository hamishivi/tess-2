import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.llama.modeling_llama import (
    LlamaForCausalLM,
    LlamaModel,
    LlamaPreTrainedModel,
)
from transformers.utils import logging

from sdlm.data.data_collator import DataCollatorForCausalLMSeq2Seq
from sdlm.data.data_utils import pad_sequence
from sdlm.models.mixins.modeling_mixin import DiffusionModelMixin

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


def get_sep_index(input_id, target):
    # TODO: improve sliding window to e.g., rolling hash
    target_length = len(target)
    for i in range(len(input_id) - len(target) + 1):
        if torch.equal(input_id[i : i + target_length], target):
            return i + target_length - 1
    raise ValueError("This is not supposed to happen")


class LlamaForSeq2SeqLM(LlamaForCausalLM):
    @torch.inference_mode()
    def generate(self, *args, **kwargs):
        context_tokens = []
        input_ids = kwargs.pop("input_ids")
        SEP = torch.tensor(
            DataCollatorForCausalLMSeq2Seq.LLAMA_SEP, device=input_ids.device
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
        outputs = super().generate(*args, **kwargs)
        seq_len = input_ids.size(1)
        output_ids = outputs[:, seq_len:]
        return output_ids.to(self.device)
