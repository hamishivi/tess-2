"""Inference script for evaluating simplex diffusion language models."""
import sys
from accelerate import Accelerator
import os
import pdb
import transformers
from sdlm.arguments import DataTrainingArguments, ModelArguments, TrainingArguments, DiffusionArguments
from transformers import HfArgumentParser, AutoConfig
from accelerate.logging import get_logger
from transformers.utils import check_min_version
from transformers.utils.versions import require_version
import datasets
import logging
from accelerate.utils import set_seed
from sdlm.utils import get_last_checkpoint
from diffusers import DDPMScheduler
from sdlm.models import RobertaForDiffusionLM

check_min_version("4.24.0")
logger = get_logger(__name__)
require_version("datasets>=1.8.0")


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
    config = AutoConfig.from_pretrained(last_checkpoint)
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=diffusion_args.num_diffusion_steps,
        beta_schedule=diffusion_args.beta_schedule,
        predict_epsilon=diffusion_args.predict_epsilon,
    )
    model = RobertaForDiffusionLM.from_pretrained(
        last_checkpoint,
        from_tf=bool(".ckpt" in last_checkpoint),
        config=config,
    )
    pipeline = 
    # TODO(rabeeh): complete this.
    texts = generate_text(inference_model=accelerator.unwrap_model(model), noise_scheduler=noise_scheduler)
    for text in texts:
        logger.info(text)


if __name__ == "__main__":
    main()
