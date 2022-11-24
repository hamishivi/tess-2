"""Training script to for simplex diffusion languge models."""
import logging
import math
import os
import random
from pathlib import Path
import sys
import numpy as np
import datasets
import torch
import pdb
import transformers
from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_from_disk
from sdlm.arguments import DataTrainingArguments, ModelArguments, TrainingArguments, DiffusionArguments
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    CONFIG_MAPPING,
    AutoConfig,
    AutoTokenizer,
    DataCollatorWithPadding,
    HfArgumentParser,
    get_scheduler,
)
from transformers.utils import check_min_version
from transformers.utils.versions import require_version
from sdlm.data.data_utils import tokenize_data, load_data, split_data_to_train_validation
from sdlm.models import RobertaForDiffusionLM
from sdlm.utils import convert_to_simplex, scale, get_norm_stats
from sdlm.schedulers import SimplexDDPMScheduler
from sdlm.pipelines.simplex_ddpm import SimplexDDPMPipeline
from eval import generate_text
import sdlm.utils as utils

# check_min_version("4.24.0")
logger = get_logger(__name__)
require_version("datasets>=1.8.0")


def save_checkpoint(args, output_dir, accelerator, model, tokenizer):
    # Removes previous checkpoints.
    utils.remove_checkpoints(args.output_dir)
    # Saves the new checkpoint.
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(output_dir, save_function=accelerator.save)
    accelerator.save_state(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info("Saving checkpoint is complete.")


def setup_logging(accelerator, logging_dir):
    logging_dir = Path(logging_dir)
    logging_dir.mkdir(exist_ok=True)
    filename = f"debug_{accelerator.process_index}.log"
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[logging.FileHandler(logging_dir / filename), logging.StreamHandler()],
    )
    if accelerator.is_main_process:  # we only want to setup logging once
        logger.setLevel(logging.INFO)
        datasets.utils.logging.set_verbosity_info()
        transformers.utils.logging.set_verbosity_info()
    else:
        logger.setLevel(logging.ERROR)
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
    return logger


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments, DiffusionArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args, diffusion_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args, diffusion_args = parser.parse_args_into_dataclasses()

    # Initialize the accelerator.
    accelerator = Accelerator(
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
        # mixed_precision=training_args.mixed_precision,
        log_with="tensorboard",
        logging_dir=f"{training_args.output_dir}/log",
    )

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("train-text-diffusion")
    logger = setup_logging(accelerator, training_args.output_dir)

    # If passed along, set the training seed now.
    if training_args.seed is not None:
        set_seed(training_args.seed)

    if accelerator.is_main_process:
        if training_args.output_dir is not None:
            os.makedirs(training_args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    if data_args.tokenized_data_path:
        tokenized_datasets = load_from_disk(data_args.tokenized_data_path)
        # TODO(rabeeh): this can take time for a large data, and we need to do it once.
        if "validation" not in tokenized_datasets:
            tokenized_datasets = split_data_to_train_validation(data_args, tokenized_datasets, training_args.seed)
    else:
        raw_datasets = load_data(data_args)
        if "validation" not in raw_datasets:
            raw_datasets = split_data_to_train_validation(data_args, raw_datasets, training_args.seed)

    # Load pretrained model and tokenizer
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, use_fast=model_args.use_fast_tokenizer)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, use_fast=model_args.use_fast_tokenizer)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    if model_args.model_name_or_path:
        model = RobertaForDiffusionLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
        )
    else:
        logger.info("Training new model from scratch")
        model = RobertaForDiffusionLM.from_config(config)

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    vocab_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > vocab_size:
        model.resize_token_embeddings(len(tokenizer))

    if not data_args.tokenized_data_path:
        tokenized_datasets = tokenize_data(data_args, tokenizer, raw_datasets, accelerator)

    train_dataset = tokenized_datasets["train"]
    eval_dataset = tokenized_datasets["validation"]
    # TODO(rabeeh): we need to add max_train samples for the non-tokenized examples with two splits as well.

    # Conditional for small test subsets
    if len(train_dataset) > 3:
        # Log a few random samples from the training set:
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # Data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, max_length=data_args.max_seq_length)

    # DataLoaders creation:
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=data_collator,
        batch_size=training_args.per_device_train_batch_size,
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        collate_fn=data_collator,
        batch_size=training_args.per_device_eval_batch_size,
    )
    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight", "timestep_embed.weight", "timestep_embed.bias"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": training_args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=training_args.learning_rate)

    # Note -> the training dataloader needs to be prepared before we grab his length below (cause its length will be
    # shorter in multiprocess)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / training_args.gradient_accumulation_steps)
    if training_args.max_train_steps is None:
        training_args.max_train_steps = training_args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        name=training_args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=training_args.num_warmup_steps * training_args.gradient_accumulation_steps,
        num_training_steps=training_args.max_train_steps * training_args.gradient_accumulation_steps
    )
    noise_scheduler = SimplexDDPMScheduler(
        num_train_timesteps=diffusion_args.num_diffusion_steps,
        beta_schedule=diffusion_args.beta_schedule,
        simplex_value=diffusion_args.simplex_value,
        clip_sample=diffusion_args.clip_sample,
        device=accelerator.device
    )
    inference_noise_scheduler = SimplexDDPMScheduler(
        num_train_timesteps=diffusion_args.num_inference_diffusion_steps,
        beta_schedule=diffusion_args.beta_schedule,
        simplex_value=diffusion_args.simplex_value,
        clip_sample=diffusion_args.clip_sample,
        device=accelerator.device
    )

    # Prepare everything with our `accelerator`.
    (model, optimizer, train_dataloader, eval_dataloader, lr_scheduler, noise_scheduler, inference_noise_scheduler) = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler, noise_scheduler, inference_noise_scheduler
    )

    # On TPU, the tie weights in our model have been disconnected, so we need to restore the ties.
    # if accelerator.distributed_type == DistributedType.TPU:
    #    model.tie_weights()

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / training_args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        training_args.max_train_steps = training_args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    training_args.num_train_epochs = math.ceil(training_args.max_train_steps / num_update_steps_per_epoch)

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = training_args.checkpointing_steps

    # Train!
    total_batch_size = (
        training_args.per_device_train_batch_size * accelerator.num_processes * training_args.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {training_args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {training_args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {training_args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {training_args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(training_args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0

    # Potentially load in the weights and states from a previous save
    if training_args.resume_from_checkpoint:
        if training_args.resume_from_checkpoint is not None or training_args.resume_from_checkpoint != "":
            accelerator.print(f"Resumed from checkpoint: {training_args.resume_from_checkpoint}")
            accelerator.load_state(training_args.resume_from_checkpoint)
            path = os.path.basename(training_args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
        # Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(path)[0]
        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
        else:
            # need to multiply `gradient_accumulation_steps` to reflect real steps
            resume_step = int(training_difference.replace("step_", "")) * training_args.gradient_accumulation_steps
            starting_epoch = resume_step // len(train_dataloader)
            resume_step -= starting_epoch * len(train_dataloader)

    # update the progress_bar if load from checkpoint
    progress_bar.update(starting_epoch * num_update_steps_per_epoch)
    completed_steps = starting_epoch * num_update_steps_per_epoch

    for epoch in range(starting_epoch, training_args.num_train_epochs):
        model.train()
        train_losses = []
        for step, batch in enumerate(train_dataloader):
            # We need to skip steps until we reach the resumed step
            if training_args.resume_from_checkpoint and epoch == starting_epoch:
                if resume_step is not None and step < resume_step:
                    if step % training_args.gradient_accumulation_steps == 0:
                        progress_bar.update(1)
                        completed_steps += 1
                    continue

            with accelerator.accumulate(model):
                # TODO(rabeeh): we need to modify this block.
                # Converts embeddings to a simplex representation.
                simplex = convert_to_simplex(batch["input_ids"], diffusion_args.simplex_value, vocab_size)
                noise = diffusion_args.simplex_value * torch.randn(simplex.shape, device=simplex.device, dtype=simplex.dtype)
                bsz = simplex.shape[0]
                # Sample a random timestep for each simplex token representation.
                timesteps = torch.randint(
                    0,
                    len(noise_scheduler),
                    (bsz,),
                    device=simplex.device,
                ).long()
                # Adds noise to each simplex representation accoding to the noise magnitude at
                # each timestep (Forward diffusion process).
                noisy_simplex = noise_scheduler.add_noise(simplex, noise, timesteps)
                # TODO(rabeeh): shouldn't they scale it before using scheduler? SSDLM scales here.
                timesteps = scale(timesteps, len(noise_scheduler))
                outputs = model(simplex=noisy_simplex, timesteps=timesteps, input_ids=batch["input_ids"])
                loss = outputs.loss
                # Keeping track of training loss for each duration of checkpointing.
                train_losses.append(loss.item())
                accelerator.backward(loss)
                norm_stats = get_norm_stats(accelerator.unwrap_model(model))
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), training_args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1

            # Logs metric every step.
            logs = {
                "train_loss": loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
                "step": completed_steps,
                **norm_stats
            }
            progress_bar.set_postfix(**logs)
            if accelerator.is_main_process:
                accelerator.log(logs, step=completed_steps)

            # Saves a checkpoint every checkpoint steps or at the end of training phase.
            if (completed_steps % checkpointing_steps == 0 or completed_steps == training_args.max_train_steps) and completed_steps != 0:
                output_dir = f"step_{completed_steps}"
                if training_args.output_dir is not None:
                    output_dir = os.path.join(training_args.output_dir, output_dir)
                '''
                accelerator.log(
                    {
                        "train_loss": np.mean(train_losses),
                        # "epoch": epoch,
                        "step": completed_steps,
                    },
                    step=completed_steps,
                )
                '''
                # TODO(rabeeh): we need to add the metrics here.
                train_losses = []
                # generates samples.
                if accelerator.is_main_process:
                    logger.info("Generating sample texts and evaluating the generated texts.")
                    pipeline = SimplexDDPMPipeline(
                        model=accelerator.unwrap_model(model),
                        scheduler=inference_noise_scheduler,
                        simplex_value=diffusion_args.simplex_value,
                        top_p=diffusion_args.top_p,
                        sampling_type=diffusion_args.sampling_type
                        #num_inference_diffusion_steps = diffusion_args.num_inference_diffusion_steps
                    )
                    results = generate_text(pipeline, tokenizer, diffusion_args, training_args, data_args)
                    for i, (pred_text_logits, pred_text_simplex) in enumerate(zip(results["pred_texts_from_logits"], results["pred_texts_from_simplex"])):
                        total_text = "*** pred_text_from_logits ***: " + pred_text_logits + "  \n"
                        total_text += "*** pred_text_from_simplex ***: " + pred_text_simplex + "  \n"
                        accelerator.trackers[0].writer.add_text(f"sample_{i}", total_text, completed_steps)
                        logger.info(total_text)
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    save_checkpoint(training_args, output_dir, accelerator, model, tokenizer)

            if completed_steps >= training_args.max_train_steps:
                break

    accelerator.end_training()


if __name__ == "__main__":
    main()
