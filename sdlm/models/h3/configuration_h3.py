from transformers.models.gpt2 import GPT2Config


class H3Config(GPT2Config):
    def __init__(
        self,
        # n_layer: int,
        # vocab_size: int,
        # max_position_embeddings=0,
        # dropout_cls=nn.Dropout,
        d_model: int = 1,
        d_inner: int = 1,
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
        # alias GPT2Config and BERTConfig
        self.layer_norm_eps = self.layer_norm_epsilon
        self.hidden_dropout_prob = self.resid_pdrop
        self.hidden_act = self.activation_function
        self.layer_norm_eps = self.layer_norm_epsilon
        # h3
        self.d_model = d_model
        self.d_inner = d_inner
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
        in_channels: int = 16,
        out_channels: int = 16,
        model_channels: int = 16,
        time_embedding_type: str = "positional",
        freq_shift: int = 0,
        flip_sin_to_cos: bool = True,
        use_pretrained_embeds: bool = False,
        use_pretrained_lm_head: bool = False,
        self_condition: bool = False,
        scale_word_embed: bool = False,
        embed_initial_std: float = None,
        tokenizer_type: str = "byte-bpe",
        add_lm_head_transform: bool = False,
        pre_layer_norm: bool = False,
        embed_layer_norm: bool = True,
        time_embed_expansion=4,
        scale_lm_head=False,
        linear_in_out_embed_projections=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        # Diffusion arguments.
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.model_channels = model_channels
        self.time_embedding_type = time_embedding_type
        self.freq_shift = freq_shift
        self.flip_sin_to_cos = flip_sin_to_cos
        self.use_pretrained_embeds = use_pretrained_embeds
        self.use_pretrained_lm_head = use_pretrained_lm_head
        self.self_condition = self_condition
        self.scale_word_embed = scale_word_embed
        self.embed_initial_std = embed_initial_std
        self.time_embed_expansion = time_embed_expansion
        # Extra model parameters.
        self.add_lm_head_transform = add_lm_head_transform
        self.pre_layer_norm = pre_layer_norm
        self.embed_layer_norm = embed_layer_norm
        self.tokenizer_type = tokenizer_type
        self.scale_lm_head = scale_lm_head
        self.linear_in_out_embed_projections = linear_in_out_embed_projections
