import logging

import gradio as gr
import torch
from transformers import (
    MODEL_FOR_MASKED_LM_MAPPING,
)

from sdlm.arguments import get_args
from sdlm.models.utils import load_model
from sdlm.pipelines.simplex_ddpm import SimplexDDPMPipeline
from sdlm.schedulers import TokenWiseSimplexDDPMScheduler

logger = logging.getLogger(__name__)
MODEL_CONFIG_CLASSES = list(MODEL_FOR_MASKED_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def main():
    model_args, data_args, training_args, diffusion_args = get_args()
    tokenizer, model = load_model(model_args, data_args, training_args, diffusion_args, logger)

    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pipeline = SimplexDDPMPipeline(
        model=model.to(device),
        scheduler=TokenWiseSimplexDDPMScheduler(
            num_train_timesteps=diffusion_args.num_train_timesteps
            if hasattr(diffusion_args, "num_train_timesteps") else 100,
            beta_schedule=getattr(diffusion_args, "beta_schedule", "squaredcos_improved_ddpm"),
            simplex_value=getattr(diffusion_args, "simplex_value", 5.0),
            clip_sample=getattr(diffusion_args, "clip_sample", False),
            device=device,
        ),
        simplex_value=getattr(diffusion_args, "simplex_value", 5.0),
        top_p=getattr(diffusion_args, "top_p", 0.99),
        sampling_type="top_p",
        is_conditional_generation=True,
        tokenizer=tokenizer,
        classifier_free_uncond_input="empty_token",
        temperature=getattr(diffusion_args, "temperature", 1.0),
        guidance_softmax_combination=True,
    )

    def generate(
        inputs,
        simplex_value=5.0,
        top_p=0.99,
        temperature=1.0,
        diffusion_steps=100,
        beta_schedule="squaredcos_improved_ddpm",
        clip_sample=False,
        guidance_scale=1.0,
        generated_sequence_length=256,
        progress=gr.Progress(),
    ):
        """
        Gradio-friendly generation function. Adjusts the pipeline's parameters
        (simplex_value, top_p, etc.) as requested, then runs generation.
        """
        with torch.inference_mode():
            # Update pipeline scheduler with user-provided parameters:
            pipeline.scheduler.num_train_timesteps = diffusion_steps
            pipeline.scheduler.beta_schedule = beta_schedule
            pipeline.scheduler.simplex_value = simplex_value
            pipeline.scheduler.clip_sample = clip_sample
            pipeline.simplex_value = simplex_value
            pipeline.top_p = top_p
            pipeline.temperature = temperature
            # tulu chat template
            inputs = "<|user|>\n" + inputs + "<|assistant|>\n"

            # Tokenize and prepare input for diffusion
            tokenized_input = tokenizer([inputs], add_special_tokens=False, return_tensors="pt").input_ids
            tokenized_input_len = tokenized_input.shape[1]

            # Concatenate BOS + input + blank space for generation
            tokenized_input = torch.cat(
                [
                    torch.ones((1, 1), dtype=torch.long) * tokenizer.bos_token_id,
                    tokenized_input,
                    torch.ones((1, generated_sequence_length), dtype=torch.long) * tokenizer.pad_token_id,
                ],
                dim=-1,
            )

            # Create a mask over the generation region
            span_mask = torch.cat(
                [
                    torch.zeros((1, tokenized_input_len + 1), dtype=torch.bool),
                    torch.ones((1, generated_sequence_length), dtype=torch.bool),
                ],
                dim=-1,
            )

            batch = {
                "input_ids": tokenized_input.to(device),
                "span_mask": span_mask.to(device),
            }

            # Run sampling
            
            pipe = pipeline(batch=batch, seq_length=generated_sequence_length, guidance_scale=guidance_scale)
            for out in pipe:
                output_ids = out.logits.argmax(dim=-1)
                generated_tokens = output_ids[:, tokenized_input_len + 1 :]
                yield tokenizer.decode(generated_tokens[0], skip_special_tokens=True)

    # Quick test call (uncomment if you want a quick, non-Gradio test)
    print("Test generation: ", generate("The best things in life are"))

    demo = gr.Interface(
        fn=generate,
        inputs=[
            gr.Textbox(lines=5, label="Input Prompt"),
            gr.Number(value=5.0, label="Simplex value"),
            gr.Slider(0, 1, value=0.99, step=0.01, label="Top-p"),
            gr.Slider(0, 5, value=1.0, step=0.1, label="Temperature"),
            gr.Number(value=100, precision=0, label="Diffusion steps"),
            gr.Dropdown(
                choices=["linear", "scaled_linear", "squaredcos_cap_v2", "squaredcos_improved_ddpm"],
                value="squaredcos_improved_ddpm",
                label="Beta schedule",
            ),
            gr.Checkbox(value=False, label="Clip sample?"),
            gr.Number(value=1.0, label="Guidance scale"),
            gr.Number(value=256, label="Generation length (tokens)"),
        ],
        outputs="text",
        title="Simplex Diffusion LM",
        description="Generate text using a simplex-based diffusion model.",
    )

    demo.queue().launch(server_name="0.0.0.0", server_port=8888, share=True)


if __name__ == "__main__":
    main()