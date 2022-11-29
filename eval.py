"""Inference script for evaluating simplex diffusion language models."""
import sys
from accelerate import Accelerator
import os
import torch
import pdb
import transformers
from sdlm.arguments import DataTrainingArguments, ModelArguments, TrainingArguments, DiffusionArguments
from transformers import HfArgumentParser, AutoTokenizer
from accelerate.logging import get_logger
from transformers.utils import check_min_version
from transformers.utils.versions import require_version
import datasets
import logging
from accelerate.utils import set_seed
from sdlm.utils import get_last_checkpoint
from sdlm.schedulers import SimplexDDPMScheduler
from sdlm.models import RobertaForDiffusionLM
from sdlm.pipelines.simplex_ddpm import SimplexDDPMPipeline
import torch.nn.functional as F
from sdlm.inference.inference_utils import process_text, split_into_masked_and_unmasked, concatenate_alternatively
from sdlm.models.configuration import RobertaDiffusionConfig

# check_min_version("4.24.0")
logger = get_logger(__name__)
require_version("datasets>=1.8.0")


def predict_conditional_generated(batch, tokenizer, predicted_token_ids, prefix_name):
    masked = list(
        map(lambda x, y: split_into_masked_and_unmasked(x, y, return_masked=True), predicted_token_ids, batch["span_mask"])
    )
    unmasked = list(
        map(lambda x, y: split_into_masked_and_unmasked(x, y, return_masked=False), batch["input_ids"], batch["span_mask"])
    )
    pred_masked_texts = [tokenizer.batch_decode(x, skip_special_tokens=False) for x in masked]
    pred_unmasked_texts = [tokenizer.batch_decode(x, skip_special_tokens=False) for x in unmasked]
    pred_texts = list(map(lambda x, y: concatenate_alternatively(x, y), pred_unmasked_texts, pred_masked_texts))
    pred_texts_marked = list(
        map(lambda x, y: concatenate_alternatively(x, y, mark="***"), pred_unmasked_texts, pred_masked_texts)
    )
    return {prefix_name: pred_texts, prefix_name + "_marked": pred_texts_marked}


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments, DiffusionArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args, diffusion_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args, diffusion_args = parser.parse_args_into_dataclasses()

    # Initialize the accelerator.
    accelerator = Accelerator(
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
        log_with="tensorboard",
        logging_dir=training_args.output_dir,
    )
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
    if training_args.seed is not None:
        set_seed(training_args.seed)
    last_checkpoint = get_last_checkpoint(training_args.output_dir, prefix_checkpoint_dir="step")
    config = RobertaDiffusionConfig.from_pretrained(last_checkpoint, self_conditioning=diffusion_args.self_conditioning)
    noise_scheduler = SimplexDDPMScheduler(
        num_train_timesteps=diffusion_args.num_inference_diffusion_steps,
        beta_schedule=diffusion_args.beta_schedule,
        simplex_value=diffusion_args.simplex_value,
        clip_sample=diffusion_args.clip_sample,
    )
    model = RobertaForDiffusionLM.from_pretrained(
        last_checkpoint,
        from_tf=bool(".ckpt" in last_checkpoint),
        config=config,
    )
    pipeline = SimplexDDPMPipeline(
        model=accelerator.unwrap_model(model),
        scheduler=noise_scheduler,
        simplex_value=diffusion_args.simplex_value,
        top_p=diffusion_args.top_p,
        sampling_type=diffusion_args.sampling_type,
        span_infilling=data_args.span_infilling,
    )
    tokenizer = AutoTokenizer.from_pretrained(last_checkpoint, use_fast=model_args.use_fast_tokenizer)
    (model, tokenizer, pipeline, noise_scheduler) = accelerator.prepare(model, tokenizer, pipeline, noise_scheduler)

    results = generate_text(pipeline, tokenizer, diffusion_args, training_args, data_args)

    if accelerator.is_main_process:
        for i, (pred_text_logits, pred_text_simplex) in enumerate(
            zip(results["pred_texts_from_logits"], results["pred_texts_from_simplex"])
        ):
            total_text = "*** pred_text_from_logits ***: " + pred_text_logits + "  \n"
            total_text += "*** pred_text_from_simplex ***: " + pred_text_simplex + "  \n"
            logger.info(total_text)


def generate_text(pipeline, tokenizer, diffusion_args, training_args, data_args, accelerator, batch=None):
    simplex = pipeline(batch_size=training_args.per_device_eval_batch_size, seq_length=data_args.max_seq_length, batch=batch)
    # Gathers results.
    simplex_results = accelerator.gather(simplex.simplex)
    logits_results = accelerator.gather(simplex.logits)
    probabilities = F.softmax(simplex_results, dim=-1)
    token_ids_from_simplex = torch.argmax(probabilities, dim=-1)
    token_ids_from_logits = torch.argmax(logits_results, dim=-1)

    results = {}
    if data_args.span_infilling:
        # We predict the masked tokens only. Here, we compute the masked tokens.
        results.update(predict_conditional_generated(batch, tokenizer, token_ids_from_simplex, "pred_texts_from_simplex"))
        results.update(predict_conditional_generated(batch, tokenizer, token_ids_from_logits, "pred_texts_from_logits"))
    else:
        results.update(
            {"pred_texts_from_simplex": tokenizer.batch_decode(token_ids_from_simplex, skip_special_tokens=False)}
        )
        results.update({"pred_texts_from_logits": tokenizer.batch_decode(token_ids_from_logits, skip_special_tokens=False)})

    return results
    # {k: process_text(v) for k, v in results.items()}


if __name__ == "__main__":
    main()
