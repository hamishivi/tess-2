from transformers import AutoTokenizer

from .cdcd.positionwise_warper_model import (
    PositionwiseCDCDRobertaConfig,
    PositionwiseCDCDRobertaForDiffusionLM,
)
from .cdcd.tokenwise_warper_model import TokenwiseCDCDRobertaForDiffusionLM
from .cdcd.warper_model import CDCDRobertaConfig, CDCDRobertaForDiffusionLM
from .roberta.configuration_roberta import RobertaDiffusionConfig
from .roberta.modeling_roberta import RobertaForDiffusionLM


def model_config_helper(model_name_or_path, use_model="cdcd"):
    if "roberta" in model_name_or_path and use_model == "cdcd":
        return CDCDRobertaConfig, CDCDRobertaForDiffusionLM
    elif "roberta" in model_name_or_path and use_model == "tokenwise_cdcd":
        return CDCDRobertaConfig, TokenwiseCDCDRobertaForDiffusionLM
    elif "roberta" in model_name_or_path and use_model == "positionwise_cdcd":
        return PositionwiseCDCDRobertaConfig, PositionwiseCDCDRobertaForDiffusionLM
    elif "roberta" in model_name_or_path:
        return RobertaDiffusionConfig, RobertaForDiffusionLM
    raise ValueError


def is_cdcd_check(model):
    return (
        isinstance(model, CDCDRobertaForDiffusionLM)
        or isinstance(model, TokenwiseCDCDRobertaForDiffusionLM)
        or isinstance(model, PositionwiseCDCDRobertaForDiffusionLM)
    )


def is_tokenwise_cdcd_check(model):
    return isinstance(model, TokenwiseCDCDRobertaForDiffusionLM) or isinstance(
        model, PositionwiseCDCDRobertaForDiffusionLM
    )


def load_model(model_args, diffusion_args, logger):
    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    cfg_cls, model_cls = model_config_helper(
        model_args.model_name_or_path, use_model=model_args.use_model
    )
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

    if model_args.model_name_or_path and not model_args.from_scratch:
        model = model_cls.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    else:
        logger.warning("Training new model from scratch")
        model = model_cls._from_config(config)
        model.init_weights()

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    vocab_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > vocab_size:
        model.resize_token_embeddings(len(tokenizer))

    return tokenizer, model
