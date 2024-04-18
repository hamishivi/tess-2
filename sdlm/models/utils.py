from typing import Optional

import torch
from transformers import AutoTokenizer

from .ar_warp.ar_warper import GARDiffusionLM
from .cdcd.ar_warper import CDCDGARRobertaForDiffusionLM
from .cdcd.positionwise_warper_model import (
    PositionwiseCDCDRobertaConfig,
    PositionwiseCDCDRobertaForDiffusionLM,
)
from .cdcd.tokenwise_warper_model import TokenwiseCDCDRobertaForDiffusionLM
from .cdcd.warper_model import CDCDRobertaConfig, CDCDRobertaForDiffusionLM
from .confidence_tracker.confidence_tracker_model import (
    ConfidenceTrackerRobertaDiffusionLM,
)
from .llama.configuration_llama import LlamaDiffusionConfig
from .llama.modeling_llama import LlamaForDiffusionLM, LlamaForSeq2SeqLM
from .roberta.configuration_roberta import RobertaDiffusionConfig
from .roberta.modeling_roberta import RobertaForDiffusionLM
from .mistral.configuration_mistral import MistralDiffusionConfig
from .mistral.modeling_mistral import MistralForDiffusionLM, MistralForSeq2SeqLM


def model_config_helper(
    model_name_or_path: str,
    use_model: str = "cdcd",
    is_diffusion: bool = True,
    conditional_generation: Optional[str] = None,
):
    if "llama" in model_name_or_path.lower():
        if conditional_generation == "seq2seq" and not is_diffusion:
            return LlamaDiffusionConfig, LlamaForSeq2SeqLM
        return LlamaDiffusionConfig, LlamaForDiffusionLM
    if "mistral" in model_name_or_path.lower():
        if conditional_generation == "seq2seq" and not is_diffusion:
            return MistralDiffusionConfig, MistralForSeq2SeqLM
        return MistralDiffusionConfig, MistralForDiffusionLM
    if "roberta" in model_name_or_path and use_model == "cdcd":
        return CDCDRobertaConfig, CDCDRobertaForDiffusionLM
    elif "roberta" in model_name_or_path and use_model == "tokenwise_cdcd":
        return CDCDRobertaConfig, TokenwiseCDCDRobertaForDiffusionLM
    elif "roberta" in model_name_or_path and use_model == "positionwise_cdcd":
        return PositionwiseCDCDRobertaConfig, PositionwiseCDCDRobertaForDiffusionLM
    elif "roberta" in model_name_or_path and use_model == "confidence":
        return RobertaDiffusionConfig, ConfidenceTrackerRobertaDiffusionLM
    elif "roberta" in model_name_or_path and use_model == "gar":
        print(
            f"Using RobertaDiffusionConfig and RobertaForDiffusionLM for {model_name_or_path}"
        )
        return RobertaDiffusionConfig, GARDiffusionLM
    elif "roberta" in model_name_or_path and use_model == "cdcdgar":
        return CDCDRobertaConfig, CDCDGARRobertaForDiffusionLM
    else:  # "roberta" in model_name_or_path:
        print(
            f"Using RobertaDiffusionConfig and RobertaForDiffusionLM for {model_name_or_path}"
        )
        return RobertaDiffusionConfig, RobertaForDiffusionLM
    raise ValueError("Unsupported model.")


def is_cdcd_check(model):
    return (
        isinstance(model, CDCDRobertaForDiffusionLM)
        or isinstance(model, TokenwiseCDCDRobertaForDiffusionLM)
        or isinstance(model, PositionwiseCDCDRobertaForDiffusionLM)
        or isinstance(model, GARDiffusionLM)
        or isinstance(model, CDCDGARRobertaForDiffusionLM)
    )


def is_tokenwise_cdcd_check(model):
    return isinstance(model, TokenwiseCDCDRobertaForDiffusionLM) or isinstance(
        model, PositionwiseCDCDRobertaForDiffusionLM
    )


def load_model(model_args, data_args, training_args, diffusion_args, logger):
    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    cfg_cls, model_cls = model_config_helper(
        model_args.model_name_or_path,
        use_model=model_args.use_model,
        is_diffusion=diffusion_args.num_diffusion_steps > 0,
        conditional_generation=data_args.conditional_generation,
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
        is_causal=model_args.is_causal,
        mask_padding_in_loss=training_args.mask_padding_in_loss,
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

    tokenizer.padding_side == "right"
    try:
        tokenizer.add_eos_token = True
    except AttributeError:
        # roberta does not have this
        pass

    if model_args.model_name_or_path and not model_args.from_scratch:
        # identify dtype
        torch_dtype = torch.float32
        if training_args.bf16:
            torch_dtype = torch.bfloat16
        elif training_args.fp16:
            torch_dtype = torch.float16
        model = model_cls.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
            torch_dtype=torch_dtype,
            attn_implementation="flash_attention_2"
            if model_args.use_flash_attention2
            else "eager",
        ).to("cuda")
    else:
        logger.warning("Training new model from scratch")
        model = model_cls._from_config(config)
        model.init_weights()

    if not tokenizer.pad_token_id:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    vocab_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > vocab_size:
        model.resize_token_embeddings(len(tokenizer))
        model.config.pad_token_id = tokenizer.pad_token_id

    return tokenizer, model
