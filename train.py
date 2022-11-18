"""Training script to for simplex diffusion languge models."""
import logging
import math
import os
import random
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
from sdlm.utils import convert_to_simplex, scale
from diffusers import DDPMScheduler

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.24.0")

logger = get_logger(__name__)
require_version(
    "datasets>=1.8.0",
    "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt",
)


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

    # If passed along, set the training seed now.
    if training_args.seed is not None:
        set_seed(training_args.seed)

    # Handle the repository creation
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
        num_training_steps=training_args.max_train_steps * training_args.gradient_accumulation_steps,
    )
    # TODO: we need to check how this works.
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=diffusion_args.num_diffusion_steps, beta_schedule=diffusion_args.beta_schedule
    )

    # Prepare everything with our `accelerator`.
    (model, optimizer, train_dataloader, eval_dataloader, lr_scheduler, noise_scheduler) = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler, noise_scheduler
    )

    # On TPU, the tie weights in our model have been disconnected, so we need to restore the ties.
    if accelerator.distributed_type == DistributedType.TPU:
        model.tie_weights()

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / training_args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        training_args.max_train_steps = training_args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    training_args.num_train_epochs = math.ceil(training_args.max_train_steps / num_update_steps_per_epoch)

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = training_args.checkpointing_steps

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    experiment_config = {**vars(model_args), **vars(training_args), **vars(diffusion_args), **vars(data_args)}
    # TensorBoard cannot log Enums, need the raw value
    experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
    accelerator.init_trackers("mlm_no_trainer", experiment_config)

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
                noise = diffusion_args.simplex_value * torch.randn(simplex.shape).to(simplex.device)
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
                # TODO(rabeeh): shouldn't they scale it before using scheduler?
                timesteps = scale(timesteps, len(noise_scheduler))
                outputs = model(inputs_embeds=noisy_simplex, timesteps=timesteps, input_ids=batch["input_ids"])
                loss = outputs.loss
                # Keeping track of training loss for each duration of checkpointing.
                train_losses.append(loss.detach().float())
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1

            # Saves a checkpoint every checkpoint steps or at the end of training phase.
            if completed_steps % checkpointing_steps == 0 or completed_steps == training_args.max_train_steps:
                output_dir = f"step_{completed_steps}"
                if training_args.output_dir is not None:
                    output_dir = os.path.join(training_args.output_dir, output_dir)
                accelerator.save_state(output_dir)
                accelerator.log(
                    {
                        "train_loss": np.mean(train_losses),
                        "epoch": epoch,
                        "step": completed_steps,
                    },
                    step=completed_steps,
                )
                train_losses = []
                accelerator.wait_for_everyone()
                unwrapped_model = accelerator.unwrap_model(model)
                # TODO(rabeeh): check resume and we need to read from the last checkpoint.
                unwrapped_model.save_pretrained(
                    output_dir,
                    is_main_process=accelerator.is_main_process,
                    save_function=accelerator.save,
                )
                if accelerator.is_main_process:
                    tokenizer.save_pretrained(output_dir)

            if completed_steps >= training_args.max_train_steps:
                break

    accelerator.end_training()


if __name__ == "__main__":
    main()
