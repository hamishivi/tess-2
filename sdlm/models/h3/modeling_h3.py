# Copyright (c) 2023, Tri Dao, Dan Fu.
import math
import random
import re
from collections import OrderedDict
from functools import partial
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from flash_attn.modules.block import Block
from flash_attn.modules.mha import MHA
from flash_attn.modules.mlp import FusedMLP, Mlp
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import MaskedLMOutput
from transformers.models.gpt2.modeling_gpt2 import GPT2PreTrainedModel
from transformers.utils import logging

try:
    from flash_attn.ops.layer_norm import dropout_add_layer_norm
except ImportError:
    dropout_add_layer_norm = None

from sdlm.utils import convert_to_simplex, mix_values_based_on_self_condition

from .ssm.h3 import H3

logger = logging.get_logger(__name__)


def create_mixer_cls(
    ssm_cls=H3, ssm_cfg=None, attn_layer_idx=None, attn_cfg=None, layer_idx=None
):
    if attn_layer_idx is not None and layer_idx in attn_layer_idx:
        causal = True if attn_cfg is None else attn_cfg.pop("causal", True)
        mixer_cls = partial(
            MHA,
            layer_idx=layer_idx,
            causal=causal,
            **(attn_cfg if attn_cfg is not None else {}),
        )
    else:
        mixer_cls = partial(
            ssm_cls, layer_idx=layer_idx, **(ssm_cfg if ssm_cfg is not None else {})
        )
    return mixer_cls


def create_mlp_cls(d_model, d_inner=None, fused_mlp=False):
    inner_dim = d_inner if d_inner is not None else 4 * d_model
    if not fused_mlp:
        mlp_cls = partial(
            Mlp,
            hidden_features=inner_dim,
            activation=partial(F.gelu, approximate="tanh"),
        )
    else:
        mlp_cls = partial(FusedMLP, hidden_features=inner_dim)
    return mlp_cls


def create_block(
    d_model,
    d_inner=None,
    ssm_cls=H3,
    ssm_cfg=None,
    attn_layer_idx=None,
    attn_cfg=None,
    layer_norm_epsilon=1e-5,
    resid_dropout1=0.0,
    resid_dropout2=0.0,
    residual_in_fp32=False,
    fused_mlp=False,
    fused_dropout_add_ln=False,
    layer_idx=None,
):
    mixer_cls = create_mixer_cls(
        ssm_cls=ssm_cls,
        ssm_cfg=ssm_cfg,
        attn_layer_idx=attn_layer_idx,
        attn_cfg=attn_cfg,
        layer_idx=layer_idx,
    )
    mlp_cls = create_mlp_cls(d_model, d_inner=d_inner, fused_mlp=fused_mlp)
    norm_cls = partial(nn.LayerNorm, eps=layer_norm_epsilon)
    block = Block(
        d_model,
        mixer_cls,
        mlp_cls,
        norm_cls=norm_cls,
        prenorm=True,
        resid_dropout1=resid_dropout1,
        resid_dropout2=resid_dropout2,
        fused_dropout_add_ln=fused_dropout_add_ln,
        residual_in_fp32=residual_in_fp32,
    )
    block.layer_idx = layer_idx
    return block


# https://github.com/huggingface/transformers/blob/c28d04e9e252a1a099944e325685f14d242ecdcd/src/transformers/models/gpt2/modeling_gpt2.py#L454
def _init_weights(
    module,
    n_layer,
    initializer_range=0.02,
    rescale_prenorm_residual=True,
    glu_act=False,
):
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, std=initializer_range)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                nn.init.normal_(
                    p, mean=0.0, std=initializer_range / math.sqrt(2 * n_layer)
                )
            # If using GLU activation for now, we scale the std by 2
            elif name in ["output_linear.0.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                if not glu_act:
                    nn.init.normal_(
                        p, mean=0.0, std=initializer_range / math.sqrt(2 * n_layer)
                    )
                else:
                    out_features = p.shape[0]
                    # Multiplying the first half of the matrix by 2 since sigmoid scales it down by 0.5
                    # on average.
                    nn.init.normal_(
                        p[: out_features // 2],
                        mean=0.0,
                        std=initializer_range / math.sqrt(2 * n_layer) * 2,
                    )


class SSMModel(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        # self.embeddings = GPT2Embeddings(d_model, vocab_size, max_position_embeddings)
        self.residual_in_fp32 = config.residual_in_fp32

        # We change the order of dropout, residual and layer norm:
        # Instead of LN -> Attn / MLP -> Dropout -> Add, we do:
        # Dropout -> Add -> LN -> Attn / MLP, returning both the residual branch (output of Add) and
        # the main branch (output of MLP). The model definition is unchanged, but the mapping of the
        # nn.Dropout probabilities are changed.
        # This is for performance reason: we can fuse dropout + add + layer_norm.
        self.fused_dropout_add_ln = config.fused_dropout_add_ln
        if self.fused_dropout_add_ln and dropout_add_layer_norm is None:
            raise ImportError("dropout_add_layer_norm is not installed")

        self.layers = nn.ModuleList(
            [
                create_block(
                    config.d_model,
                    d_inner=config.d_inner,
                    ssm_cfg=config.ssm_cfg,
                    attn_layer_idx=config.attn_layer_idx,
                    attn_cfg=config.attn_cfg,
                    layer_norm_epsilon=config.layer_norm_epsilon,
                    resid_dropout1=config.embed_dropout
                    if i == 0
                    else config.resid_dropout,
                    resid_dropout2=config.resid_dropout,
                    residual_in_fp32=config.residual_in_fp32,
                    fused_mlp=config.fused_mlp,
                    fused_dropout_add_ln=config.fused_dropout_add_ln,
                    layer_idx=i,
                )
                for i in range(config.n_layer)
            ]
        )

        self.drop_f = nn.Dropout(config.resid_dropout)
        self.ln_f = nn.LayerNorm(config.d_model, eps=config.layer_norm_epsilon)

        self.apply(
            partial(
                _init_weights,
                n_layer=config.n_layer,
                **(
                    config.initializer_cfg if config.initializer_cfg is not None else {}
                ),
            )
        )

    def forward(self, hidden_states, inference_params=None):
        # hidden_states = self.embeddings(input_ids, position_ids=position_ids)
        residual = None
        mixer_kwargs = None
        if inference_params is not None:
            mixer_kwargs = dict(inference_params=inference_params)
        for layer in self.layers:
            hidden_states, residual = layer(
                hidden_states, residual, mixer_kwargs=mixer_kwargs
            )
        if not self.fused_dropout_add_ln:
            dropped = self.drop_f(hidden_states)
            residual = (dropped + residual) if residual is not None else dropped
            hidden_states = self.ln_f(residual.to(dtype=self.ln_f.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
            hidden_states = dropout_add_layer_norm(
                hidden_states,
                residual,
                self.ln_f.weight,
                self.ln_f.bias,
                self.drop_f.p if self.training else 0.0,
                self.ln_f.eps,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )
        return hidden_states


class H3ForDiffusionLM(GPT2PreTrainedModel):
    _keys_to_ignore_on_load_missing = [
        r"attn.masked_bias",
        r"attn.bias",
        r"lm_head.weight",
    ]

    def __init__(self, config):
        super().__init__(config)
        self.h3 = SSMModel(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_h3_empty_tokens(self, shape, device):
        if self.config.empty_token_be_mask:
            empty_token_ids = (
                torch.ones(shape, dtype=torch.int64, device=device) * 50264
            )
        else:
            # TODO: check if this is true
            # Padding token in gpt2 is 1
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
        classifier_free_guidance: bool = False,
        classifier_free_guidance_in_train: bool = False,
        # unconditional_simplex: torch.FloatTensor = None,
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
        seq_length = inputs_probs.shape[1]
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
        timesteps_embed = self.timestep_embed(timesteps.view(-1, 1).float())
        inputs_embeds = inputs_embeds + timesteps_embed.unsqueeze(1).repeat(
            1, seq_length, 1
        )

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
        outputs = self.roberta(
            input_ids=None,  # TODO(rabeeh): we can remove this hack when we moved loss to outside.
            attention_mask=None,  # attention_mask,
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
            loss_fct = CrossEntropyLoss()
            labels = (
                torch.where(span_mask, input_ids, -100)
                if span_mask is not None
                else input_ids
            )
            masked_lm_loss = loss_fct(
                prediction_scores_for_loss.view(-1, self.config.vocab_size),
                labels.view(-1),
            )

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return (
                ((masked_lm_loss,) + output) if masked_lm_loss is not None else output
            )

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.last_hidden_state,
            attentions=outputs.attentions,
        )

    def resize_position_embeddings(
        self, new_num_position_embeddings: int, with_alternatation=False
    ):
        """
        Resizes position embeddings of the model if `new_num_position_embeddings != config.max_position_embeddings`.
        Arguments:
            new_num_position_embeddings (`int`):
                The number of new position embedding matrix. If position embeddings are learned, increasing the size
                will add newly initialized vectors at the end, whereas reducing the size will remove vectors from the
                end. If position embeddings are not learned (*e.g.* sinusoidal position embeddings), increasing the
                size will add correct vectors at the end following the position encoding algorithm, whereas reducing
                the size will remove vectors from the end.
        """
        num_position_embeds_diff = (
            new_num_position_embeddings - self.config.max_position_embeddings
        )

        # no resizing needs to be done if the length stays the same
        if num_position_embeds_diff == 0:
            return

        logger.info(
            f"Setting `config.max_position_embeddings={new_num_position_embeddings}`..."
        )
        self.config.max_position_embeddings = new_num_position_embeddings
        old_position_embeddings_weight = (
            self.roberta.embeddings.position_embeddings.weight.clone()
        )

        padding_idx = self.config.pad_token_id
        self.roberta.embeddings.position_embeddings = nn.Embedding(
            self.config.max_position_embeddings,
            self.config.hidden_size,
            padding_idx=padding_idx,
        )
        with torch.no_grad():
            if num_position_embeds_diff > 0:
                self.roberta.embeddings.position_embeddings.weight[
                    :-num_position_embeds_diff
                ] = nn.Parameter(old_position_embeddings_weight)
                if with_alternatation:
                    self.roberta.embeddings.position_embeddings.weight[
                        -num_position_embeds_diff:
                    ] = nn.Parameter(
                        old_position_embeddings_weight[:num_position_embeds_diff]
                    )
            else:
                self.roberta.embeddings.position_embeddings.weight = nn.Parameter(
                    old_position_embeddings_weight[:num_position_embeds_diff]
                )
        # move position_embeddings to correct device
        self.roberta.embeddings.position_embeddings.to(self.device)
        # Update other needed parameters.
        self.roberta.embeddings.position_ids = (
            torch.arange(self.config.max_position_embeddings)
            .expand((1, -1))
            .type_as(self.roberta.embeddings.position_ids)
        )
        self.roberta.embeddings.token_type_ids = torch.zeros(
            self.roberta.embeddings.position_ids.size(), dtype=torch.long
        ).type_as(self.roberta.embeddings.token_type_ids)

        # resize the distance embeddings.
        for i in range(self.config.num_hidden_layers):
            if (
                self.config.position_embedding_type == "relative_key"
                or self.config.position_embedding_type == "relative_key_query"
            ):
                self.roberta.encoder.layer[
                    i
                ].attention.self.distance_embedding = nn.Embedding(
                    2 * self.config.max_position_embeddings - 1,
                    self.attention_head_size,
                )
                old_distance_embedding_weight = self.layer[
                    i
                ].attention.self.distance_embedding.weight.clone()
                with torch.no_grad():
                    if num_position_embeds_diff > 0:
                        self.roberta.encoder.layer[
                            i
                        ].attention.self.distance_embedding.weight[
                            : -2 * num_position_embeds_diff
                        ] = nn.Parameter(
                            old_distance_embedding_weight
                        )
                    else:
                        self.roberta.encoder.layer[
                            i
                        ].attention.self.distance_embedding.weight = nn.Parameter(
                            old_distance_embedding_weight[
                                : 2 * num_position_embeds_diff
                            ]
                        )

    def load_state_dict(self, state_dict, strict=True):
        # Remapping from our checkpoints that used different names
        def key_mapping_backbone(key):
            key = re.sub(r"^s4seq.encoder.", "backbone.", key)
            key = re.sub(r"^embedding.", "backbone.embeddings.word_embeddings.", key)
            key = re.sub(r"^backbone.norm", "backbone.ln_0", key)
            return key

        state_dict = OrderedDict(
            (key_mapping_backbone(k), v) for k, v in state_dict.items()
        )
        # Remapping from our checkpoints that used a different ordering of layers in the block
        # Previous: Mixer / MLP -> Dropout -> Add -> LN
        # Current: Dropout -> Add -> LN -> Attn / MLP
        if "backbone.ln_0.weight" in state_dict:
            n_layers = len(self.backbone.layers)
            ln_weight = state_dict.pop(f"backbone.layers.{n_layers - 1}.norm2.weight")
            ln_bias = state_dict.pop(f"backbone.layers.{n_layers - 1}.norm2.bias")
            state_dict["backbone.ln_f.weight"] = ln_weight
            state_dict["backbone.ln_f.bias"] = ln_bias
            for l in reversed(range(n_layers)):  # noqa: E741
                ln_weight = state_dict.pop(f"backbone.layers.{l}.norm1.weight")
                ln_bias = state_dict.pop(f"backbone.layers.{l}.norm1.bias")
                state_dict[f"backbone.layers.{l}.norm2.weight"] = ln_weight
                state_dict[f"backbone.layers.{l}.norm2.bias"] = ln_bias
                if l > 0:
                    ln_weight = state_dict.pop(f"backbone.layers.{l - 1}.norm2.weight")
                    ln_bias = state_dict.pop(f"backbone.layers.{l - 1}.norm2.bias")
                    state_dict[f"backbone.layers.{l}.norm1.weight"] = ln_weight
                    state_dict[f"backbone.layers.{l}.norm1.bias"] = ln_bias
            ln_weight = state_dict.pop("backbone.ln_0.weight")
            ln_bias = state_dict.pop("backbone.ln_0.bias")
            state_dict["backbone.layers.0.norm1.weight"] = ln_weight
            state_dict["backbone.layers.0.norm1.bias"] = ln_bias
        return super().load_state_dict(state_dict, strict=strict)
