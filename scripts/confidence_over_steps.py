import logging
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.special import kl_div
from scipy.stats import entropy
from transformers import (
    MODEL_FOR_MASKED_LM_MAPPING,
    AutoTokenizer,
    HfArgumentParser,
    set_seed,
)

from sdlm.arguments import DiffusionArguments, ModelArguments
from sdlm.models.cdcd.tokenwise_warper_model import (
    TokenwiseCDCDRobertaConfig,
    TokenwiseCDCDRobertaForDiffusionLM,
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
    tokenizer, model = load_model(
        model_args, data_args, training_args, diffusion_args, logger
    )

    def generate(
        inputs,
        simplex_value=5.0,
        top_p=1.0,
        temperature=1.0,
        diffusion_steps=1000,
        beta_schedule="squaredcos_improved_ddpm",
        clip_sample=False,
        guidance_scale=1.0,
        generated_sequence_length=256,
    ):
        generated_sequence_length = int(generated_sequence_length)
        tokenized_input = tokenizer(
            [inputs], add_special_tokens=False, return_tensors="pt"
        ).input_ids
        tokenized_input_len = tokenized_input.shape[1]
        tokenized_input = torch.cat(
            [
                torch.ones((1, 1)),
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
            "seq_length": generated_sequence_length,
            "batch": inputs,
            "guidance_scale": guidance_scale,
            "is_generator": True,
        }
        # return the generator
        return pipeline(**pipeline_args)

    generator = generate(
        "When I talk about music, I talk about",
        generated_sequence_length=50,
        diffusion_steps=100,
    )
    confidences = []
    diff_2_confidences = []
    kl_divs = []
    entropies = []
    dists = []
    prev_dist = None
    final_toks = None
    for i, output in enumerate(generator):
        dist = torch.softmax(output.logits, dim=-1)
        dists.append(dist.cpu().detach())
        conf = dist.max(dim=-1).values
        conf_sec = dist.topk(2, dim=-1).values[:, :, 1]
        diff_2_confidences.append(conf - conf_sec)
        confidences.append(conf)
        entropies.append(entropy(dist.cpu().numpy(), axis=-1))
        if prev_dist is not None:
            kl_divs.append(kl_div(dist.cpu(), prev_dist.cpu()).mean(-1))
        prev_dist = dist
        final_toks = output.logits[0].argmax(-1).cpu()
        # if i > (800):
        #     break
    tokens = [tokenizer.decode(t) for t in output.logits[0].argmax(-1).cpu().numpy()]
    confidences = torch.cat(confidences, dim=0).cpu().numpy()
    prompt_len = len(tokenizer("When I talk about music, I talk about").input_ids)
    confidences[:, :prompt_len] = 1
    plt.figure(figsize=(15, 6))
    heatmap = plt.imshow(
        confidences,
        cmap="hot",
        interpolation="nearest",
        aspect="auto",  # norm=LogNorm(vmin=0.000000001, vmax=1)
    )
    plt.xticks(range(len(tokens)), tokens, rotation=90)
    plt.colorbar(heatmap)
    plt.xlabel("token position")
    plt.ylabel("Diffusion Step")
    plt.savefig("confidence_over_steps.png")
    plt.clf()

    tok_probs = []
    for d in dists:
        tok_probs.append(
            torch.stack(
                [d[0, i, tok_idx] for i, tok_idx in enumerate(final_toks)], dim=0
            )
        )
    confidences = torch.stack(tok_probs, dim=0).cpu().numpy()
    plt.figure(figsize=(15, 6))
    heatmap = plt.imshow(
        confidences,
        cmap="hot",
        interpolation="nearest",
        aspect="auto",  # norm=LogNorm(vmin=0.000000001, vmax=1)
    )
    plt.xticks(range(len(tokens)), tokens, rotation=90)
    plt.colorbar(heatmap)
    plt.xlabel("token position")
    plt.ylabel("diffusion step")
    plt.savefig("confidence_over_steps_final_token_prob.png")
    plt.clf()

    entropies = np.concatenate(entropies, axis=0)
    plt.figure(figsize=(15, 6))
    heatmap = plt.imshow(
        entropies, cmap="hot", interpolation="nearest", aspect="auto", vmax=1
    )
    plt.xticks(range(len(tokens)), tokens, rotation=90)
    plt.colorbar(heatmap)
    plt.xlabel("token position")
    plt.ylabel("diffusion step")
    plt.savefig("entopy_dist.png")
    plt.clf()

    diff_2_confidences = torch.cat(diff_2_confidences, dim=0).cpu().numpy()
    plt.figure(figsize=(15, 6))
    heatmap = plt.imshow(
        diff_2_confidences, cmap="hot", interpolation="nearest", aspect="auto"
    )
    plt.xticks(range(len(tokens)), tokens, rotation=90)
    plt.colorbar(heatmap)
    plt.xlabel("token position")
    plt.ylabel("diffusion step")
    plt.savefig("diff_two_confidence_over_steps.png")
    plt.clf()

    kl_divs = torch.cat(kl_divs, dim=0).cpu().numpy()
    plt.figure(figsize=(15, 6))
    heatmap = plt.imshow(kl_divs, cmap="hot", interpolation="nearest", aspect="auto")
    plt.xticks(range(len(tokens)), tokens, rotation=90)
    plt.colorbar(heatmap)
    plt.xlabel("token position")
    plt.ylabel("diffusion step")
    plt.savefig("kl_div.png")
    plt.clf()
    print(f"Prediction: {tokenizer.decode(output.logits[0].argmax(-1).cpu().numpy())}")


if __name__ == "__main__":
    main()
