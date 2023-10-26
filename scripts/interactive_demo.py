import logging
import os
import sys

import gradio as gr
import torch
from transformers import (
    MODEL_FOR_MASKED_LM_MAPPING,
    AutoTokenizer,
    HfArgumentParser,
    set_seed,
)

from sdlm.arguments import DiffusionArguments, ModelArguments
from sdlm.models import CDCDRobertaConfig, CDCDRobertaForDiffusionLM
from sdlm.pipelines.simplex_ddpm import SimplexDDPMPipeline
from sdlm.schedulers import TokenWiseSimplexDDPMScheduler

logger = logging.getLogger(__name__)
MODEL_CONFIG_CLASSES = list(MODEL_FOR_MASKED_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def main():
    parser = HfArgumentParser((ModelArguments, DiffusionArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, diffusion_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, diffusion_args = parser.parse_args_into_dataclasses()

    # Set seed before initializing model.
    set_seed(42)
    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    config = CDCDRobertaConfig.from_pretrained(
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

    if model_args.model_name_or_path:
        model = CDCDRobertaForDiffusionLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    else:
        raise RuntimeError("You need to load a pretrained model")

    # We resize the xs only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    vocab_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > vocab_size:
        model.resize_token_embeddings(len(tokenizer))

    # for some insane reason some of the model is not correctly loaded using from_pretrained...
    state_dict = torch.load(
        os.path.join(model_args.model_name_or_path, "pytorch_model.bin"),
        map_location="cpu",
    )
    # for some insane reason the word embeddings dont get loaded
    model.roberta.embeddings.word_embeddings.weight = torch.nn.Parameter(
        state_dict["roberta.embeddings.word_embeddings.weight"]
    )
    model.tie_weights()
    # make sure loading is entirely correct.
    assert (
        len(
            [k for k in state_dict if torch.any(state_dict[k] != model.state_dict()[k])]
        )
        == 0
    )

    def generate(
        inputs,
        simplex_value=5.0,
        top_p=0.99,
        temperature=1.0,
        diffusion_steps=2500,
        beta_schedule="squaredcos_improved_ddpm",
        clip_sample=False,
        guidance_scale=1.0,
        generated_sequence_length=256,
        progress=gr.Progress(),
    ):
        generated_sequence_length = int(generated_sequence_length)
        tokenized_input = tokenizer(
            [inputs], add_special_tokens=False, return_tensors="pt"
        ).input_ids
        tokenized_input_len = tokenized_input.shape[1]
        tokenized_input = torch.cat(
            [
                torch.ones((1, 1)) * tokenizer.bos_token_id,
                tokenized_input,
                torch.ones((1, generated_sequence_length)),
            ],
            axis=-1,
        ).long()
        span_mask = torch.cat(
            [
                torch.zeros((1, tokenized_input_len + 1)),
                torch.ones((1, generated_sequence_length)),
            ],
            axis=-1,
        ).bool()
        inputs = {"input_ids": tokenized_input.cuda(), "span_mask": span_mask.cuda()}

        model.eval()

        pipeline = SimplexDDPMPipeline(
            model=model.cuda(),
            scheduler=TokenWiseSimplexDDPMScheduler(
                num_train_timesteps=diffusion_steps,
                beta_schedule=beta_schedule,
                simplex_value=simplex_value,
                clip_sample=clip_sample,
                device=torch.device("cuda", 0),
            ),
            simplex_value=simplex_value,
            top_p=top_p,
            sampling_type="top_p",  # currently only this is supported
            is_conditional_generation=True,
            tokenizer=tokenizer,
            classifier_free_uncond_input="empty_token",
            temperature=temperature,
            guidance_softmax_combination=True,
        )
        # pipeline.progress_bar = progress.tqdm
        pipeline_args = {
            "batch_size": 1,
            "seq_length": generated_sequence_length,
            "batch": inputs,
            "guidance_scale": guidance_scale,
            "is_generator": True,
        }
        for i, output in enumerate(pipeline(**pipeline_args)):
            yield tokenizer.decode(output.logits.argmax(-1)[0])

    generate("The best things in life are")

    demo = gr.Interface(
        fn=generate,
        inputs=[
            gr.Textbox(lines=5),
            gr.Number(value=5.0),
            gr.Slider(0, 1, value=0.99),
            gr.Slider(0, 5, value=1),
            gr.Number(value=100, precision=0),
            gr.Dropdown(
                choices=[
                    "linear",
                    "scaled_linear",
                    "squaredcos_cap_v2",
                    "squaredcos_improved_ddpm",
                ],
                value="squaredcos_improved_ddpm",
            ),
            gr.Checkbox(value=False),
            gr.Number(value=1.0),
            gr.Number(value=256),
        ],
        outputs="text",
    )

    demo.queue().launch(server_name="0.0.0.0", server_port=8888, share=True)


if __name__ == "__main__":
    main()
