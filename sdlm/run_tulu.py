"""
Fine-tuning the library models for sequence to sequence.
Specifically for instruction tuning.
Runs alpacaEval as an intermediate set.
"""

import logging
import os
import sys
from collections import defaultdict

import datasets
import torch
import transformers
from transformers import set_seed
from transformers.trainer_callback import TrainerState
from transformers.utils import check_min_version
from transformers.utils.versions import require_version

from .arguments import get_args
from .data.data_collator import DataCollatorForMultiTurnSeq2Seq
from .data.data_utils import load_data
from .models import load_model
from .schedulers import TokenWiseSimplexDDPMScheduler
from .trainers.trainer_diffusion import DiffusionTrainer
from .utils import (
    encode_with_messages_format_v1,
    encode_with_messages_format_v2_batch,
    get_last_checkpoint_with_beaker_preemption,
    resolve_last_checkpoint_vs_resume_from_checkpoint,
)
from .data.instruction_evals.instruction_evals import EVAL_MAPPING

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.25.0")
require_version("datasets>=1.8.0")
logger = logging.getLogger(__name__)





def main():
    # parse args
    model_args, data_args, training_args, diffusion_args = get_args()

    if diffusion_args.eval_dataset_name and diffusion_args.eval_dataset_name not in EVAL_MAPPING:\
        raise ValueError(
            f"Invalid eval dataset name: {diffusion_args.eval_dataset_name}. Must be one of {list(EVAL_MAPPING.keys())}"
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = get_last_checkpoint_with_beaker_preemption(training_args)

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # load data
    raw_datasets = load_data(data_args, model_args)

    # load model
    tokenizer, model = load_model(
        model_args, data_args, training_args, diffusion_args, logger
    )
    tokenizer.add_eos_token = False  # since the chat template adds it

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    vocab_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > vocab_size:
        model.resize_token_embeddings(len(tokenizer))

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    if training_args.do_train:
        train_column_names = raw_datasets["train"].column_names
    # if training_args.do_eval:
    #     eval_column_names = eval_dataset.column_names

    # Temporarily set max_target_length for training.
    max_target_length = data_args.max_seq_length
    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
        with training_args.main_process_first(desc="train dataset map pre-processing"):
            # we assume the data is in the tulu format
            train_dataset = train_dataset.map(
                lambda x: encode_with_messages_format_v2_batch(
                    x,
                    tokenizer=tokenizer,
                    max_seq_length=max_target_length,
                    is_tulu_pair=data_args.is_tulu_pair,
                    is_tulu_multiturn=data_args.is_tulu_multiturn,
                    is_tulu_sliding_window_multiturn=data_args.is_tulu_sliding_window_multiturn,
                ),
                batched=True,
                # NOTE: uncomment to use v1
                # lambda x: encode_with_messages_format_v1(x, tokenizer, max_target_length),
                # batched=False,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
                remove_columns=train_column_names,
                desc="Running tokenizer on train dataset",
            )
            train_dataset.set_format("pt")
            train_dataset = train_dataset.filter(lambda x: (x["labels"] != -100).any())

    if training_args.do_eval:
        eval_dataset = EVAL_MAPPING[diffusion_args.eval_dataset_name].construct_eval_dataset(
            tokenizer, max_target_length, data_args.max_eval_samples
        )

    def preprocess_logits_for_metrics(logits):
        return logits.argmax(dim=-1)

    # Data collator. To be consistent with the run_mlm.py we need to add `mode`.
    data_collator = lambda mode: DataCollatorForMultiTurnSeq2Seq(  # noqa: E731
        tokenizer,
        # Note that if you do not use `pad_to_max_length`, this becomes very slow on multi-gpus.
        padding="max_length" if data_args.pad_to_max_length else True,
        max_length=data_args.max_seq_length,
        pad_to_multiple_of=8 if training_args.fp16 else None,
    )

    noise_scheduler = TokenWiseSimplexDDPMScheduler(
        num_train_timesteps=diffusion_args.num_diffusion_steps,
        beta_schedule=diffusion_args.beta_schedule,
        simplex_value=diffusion_args.simplex_value,
        clip_sample=diffusion_args.clip_sample,
        device=training_args.device,
        # multiply_factor=diffusion_args.multiply_factor,
    )
    inference_noise_schedulers = [
        TokenWiseSimplexDDPMScheduler(
            num_train_timesteps=timesteps,
            beta_schedule=diffusion_args.beta_schedule,
            simplex_value=diffusion_args.simplex_value,
            clip_sample=diffusion_args.clip_sample,
            device=training_args.device,
            # multiply_factor=diffusion_args.multiply_factor,
        )
        for timesteps in diffusion_args.num_inference_diffusion_steps
    ]

    compute_metrics = lambda x: EVAL_MAPPING[diffusion_args.eval_dataset_name].compute_metrics(x, skip_special_tokens=data_args.skip_special_tokens)  # noqa: E731

    # Initialize our Trainer
    trainer = DiffusionTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
        if (training_args.do_eval or training_args.do_predict)
        else None,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics
        if (training_args.do_eval or training_args.do_predict)
        else None,
        noise_scheduler=noise_scheduler,
        diffusion_args=diffusion_args,
        data_args=data_args,
        inference_noise_schedulers=inference_noise_schedulers,
    )

    # Training
    if training_args.do_train:
        checkpoint = resolve_last_checkpoint_vs_resume_from_checkpoint(
            last_checkpoint,
            training_args.resume_from_checkpoint,
        )
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples
            if data_args.max_train_samples is not None
            else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # We will load the best model here to avoid an issue when do_train is not set.
    if training_args.load_states_in_eval_from_model_path and not training_args.do_train:
        trainer.state = TrainerState.load_from_json(
            os.path.join(model_args.model_name_or_path, "trainer_state.json")
        )
        if (
            training_args.load_best_model_at_end
            and trainer.state.best_model_checkpoint is not None
        ):
            checkpoint_path = trainer.state.best_model_checkpoint
        else:
            checkpoint_path = model_args.model_name_or_path
        trainer._load_from_checkpoint(checkpoint_path)
        trainer._load_rng_state(checkpoint_path)

    # Evaluation
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        max_eval_samples = (
            data_args.max_eval_samples
            if data_args.max_eval_samples is not None
            else len(eval_dataset)
        )
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
    return results


if __name__ == "__main__":
    main()
