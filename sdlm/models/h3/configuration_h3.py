from typing import Optional

from transformers.models.gpt2 import GPT2Config


class H3Config(GPT2Config):
    def __init__(
        self,
        # n_layer: int,
        # vocab_size: int,
        # max_position_embeddings=0,
        # d_inner: int = 3072,
        d_model: int = 1,
        n_head: int = 1,
        rotary_emb_dim: int = 0,
        attn_layer_idx=None,
        resid_dropout: float = 0.0,
        embed_dropout: float = 0.1,
        layer_norm_epsilon: float = 1e-5,
        initializer_cfg=None,
        fused_mlp=False,
        fused_dropout_add_ln=False,
        residual_in_fp32=False,
        pad_vocab_size_multiple: int = 1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        # h3
        self.d_model = d_model
        self.d_inner = d_model * 4
        self.ssm_cfg = {"mode": "diag", "measure": "diag-lin"}
        self.attn_layer_idx = attn_layer_idx
        self.attn_cfg = {"num_heads": n_head}
        if rotary_emb_dim:
            self.attn_cfg["rotary_emb_dim"] = rotary_emb_dim
        self.resid_dropout = resid_dropout
        self.embed_dropout = embed_dropout
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_cfg = initializer_cfg
        self.fused_mlp = fused_mlp
        self.fused_dropout_add_ln = fused_dropout_add_ln
        self.residual_in_fp32 = residual_in_fp32
        self.pad_vocab_size_multiple = pad_vocab_size_multiple


class H3DiffusionConfig(H3Config):
    def __init__(
        self,
        self_condition: Optional[str] = None,
        self_condition_zeros_after_softmax: bool = False,
        deepmind_conditional: bool = False,
        classifier_free_simplex_inputs: bool = False,
        classifier_free_uncond_input: str = "empty_token",
        self_condition_mlp_projection=False,
        self_condition_mix_before_weights=False,
        self_condition_mix_logits_before_weights=False,
        empty_token_be_mask=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.self_condition = self_condition
        self.self_condition_zeros_after_softmax = self_condition_zeros_after_softmax
        self.deepmind_conditional = deepmind_conditional
        self.classifier_free_simplex_inputs = classifier_free_simplex_inputs
        self.classifier_free_uncond_input = classifier_free_uncond_input
        self.self_condition_mlp_projection = self_condition_mlp_projection
        self.self_condition_mix_before_weights = self_condition_mix_before_weights
        self.self_condition_mix_logits_before_weights = (
            self_condition_mix_logits_before_weights
        )
        self.empty_token_be_mask = empty_token_be_mask
        # PAD
        self.vocab_size += 1
