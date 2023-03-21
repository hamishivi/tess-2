from transformers import AutoTokenizer


from .longformer.configuration_longformer import LongformerDiffusionConfig
from .longformer.modeling_longformer import LongformerForDiffusionLM
from .roberta.configuration_roberta import RobertaDiffusionConfig
from .roberta.modeling_roberta import RobertaForDiffusionLM
from .xlm_roberta.configuration_xlm_roberta import XLMRobertaDiffusionConfig
from .xlm_roberta.modeling_xlm_roberta import XLMRobertaForDiffusionLM
try:
    from .h3.configuration_h3 import H3DiffusionConfig
    from .h3.modeling_h3 import H3ForDiffusionLM
except ModuleNotFoundError:
    H3ForDiffusionLM, H3DiffusionConfig = None, None
    pass # probably due to no flash attention, which is fine

def model_config_helper(model_name_or_path):
    if "roberta" in model_name_or_path:
        return RobertaDiffusionConfig, RobertaForDiffusionLM
    if "longformer" in model_name_or_path:
        return LongformerDiffusionConfig, LongformerForDiffusionLM
    if "gpt2" in model_name_or_path:
        return H3DiffusionConfig, H3ForDiffusionLM
    if "xlm" in model_name_or_path:
        return XLMRobertaDiffusionConfig, XLMRobertaForDiffusionLM
    raise ValueError


def load_model(model_args, diffusion_args, logger):
    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    cfg_cls, model_cls = model_config_helper(model_args.model_name_or_path)
    config = cfg_cls.from_pretrained(
        model_args.model_name_or_path,
        self_condition=diffusion_args.self_condition,
        self_condition_zeros_after_softmax=diffusion_args.self_condition_zeros_after_softmax,
        deepmind_conditional=diffusion_args.deepmind_conditional,
        classifier_free_simplex_inputs=diffusion_args.classifier_free_simplex_inputs,
        classifier_free_uncond_input=diffusion_args.classifier_free_uncond_input,
        self_condition_mlp_projection=diffusion_args.self_condition_mlp_projection,
        self_condition_mix_before_weights=diffusion_args.self_condition_mix_before_weights,
        self_condition_mix_logits_before_weights=diffusion_args.self_condition_mix_logits_before_weights,
        empty_token_be_mask=diffusion_args.empty_token_be_mask,
        d_model=model_args.d_model,
        n_head=model_args.n_head,
        attn_layer_idx=model_args.attn_layer_idx,
        attention_window=model_args.attention_window,
        **config_kwargs,
    )
    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name, **tokenizer_kwargs
        )
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path, **tokenizer_kwargs
        )
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )
    if not tokenizer.pad_token_id:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    if model_args.model_name_or_path:
        model = model_cls.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    else:
        logger.info("Training new model from scratch")
        model = model_cls.from_config(config)

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    vocab_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > vocab_size:
        model.resize_token_embeddings(len(tokenizer))

    return tokenizer, model
