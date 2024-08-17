""" Finetuning the library models for sequence classification on GLUE."""

import logging
import sys

import alpaca_eval
import datasets
import transformers
from datasets import load_dataset
from transformers import AutoTokenizer, set_seed

from .arguments import get_args
from .data.data_collator import DataCollatorForCausalMultiTurnSeq2Seq
from .data.data_utils import load_data
from .models import load_model
from .run_tulu import encode_with_messages_format_v1
from .trainers.trainer_ar import ARTrainer
from .utils import (
    get_last_checkpoint_with_beaker_preemption,
    resolve_last_checkpoint_vs_resume_from_checkpoint,
)

logger = logging.getLogger(__name__)


def main():
    # parse args
    model_args, data_args, training_args, diffusion_args = get_args()
    assert data_args.dataset_name is not None
    data_args.dataset_name = data_args.dataset_name.lower()

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

    # load dataset
    raw_datasets = load_data(data_args, model_args)
    eval_dataset = load_dataset("tatsu-lab/alpaca_eval")["eval"]

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # load tokenizer early
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name
        if model_args.tokenizer_name
        else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        padding_side=model_args.tokenizer_padding_side,
    )
    # load model
    tokenizer, model = load_model(
        model_args, data_args, training_args, diffusion_args, logger
    )

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
                lambda x: encode_with_messages_format_v1(
                    x, tokenizer, max_target_length
                ),
                batched=False,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
                remove_columns=train_column_names,
                desc="Running tokenizer on train dataset",
            )
            train_dataset.set_format("pt")
            train_dataset = train_dataset.filter(lambda x: (x["labels"] != -100).any())

    if training_args.do_eval:
        logger.warn(
            "Running evaluation. This calls GPT-4, so PLEASE MAKE SURE YOU ARE NOT RUNNING IT A TONNE"
        )
        max_target_length = data_args.max_seq_length
        # put the dataset into the correct format
        eval_dataset = eval_dataset.map(
            lambda x: {"messages": [{"role": "user", "content": x["instruction"]}]}
        )
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))
        with training_args.main_process_first(
            desc="validation dataset map pre-processing"
        ):
            prompt_function = lambda x: encode_with_messages_format_v1(  # noqa: E731
                x, tokenizer, max_target_length, add_generation_prompt=True
            )
            # prompting
            eval_dataset = eval_dataset.map(
                prompt_function,
                batched=False,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
                remove_columns=[
                    "instruction",
                    "dataset",
                    "generator",
                    "messages",
                    "output",
                ],
                desc="Running tokenizer on validation dataset",
            )
            eval_dataset.set_format("pt")
            eval_dataset.remove_columns(["labels"])
    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )

    # Metric
    def compute_metrics(results):
        metrics = {}
        eval_data = [
            tokenizer.decode(x, skip_special_tokens=True)
            .replace("<|user|>\n", "")
            .replace("<|assistant|>\n", "")
            .strip()
            for x in results.inputs
        ]
        # assume we stopped at eos
        decoded_preds = []
        for prediction in results.predictions:
            # sometimes we get out of range somehow?? guard against it.
            prediction = [x for x in prediction if x > 0 and x < tokenizer.vocab_size]
            decoded_preds.append(tokenizer.decode(prediction, skip_special_tokens=True))
        # for each decoded sample, format into alpacaeval setup
        decoded_preds = [
            {"output": y, "instruction": x, "generator": "tess2"}
            for x, y in zip(eval_data, decoded_preds)
        ]
        df_leaderboard, _ = alpaca_eval.evaluate(
            model_outputs=decoded_preds,
            is_overwrite_leaderboard=True,
            is_return_instead_of_print=True,
        )
        # grab tess2 results
        key_metrics = df_leaderboard.loc["tess2"].to_dict()
        metrics.update(key_metrics)
        return metrics

    # Data collator. To be consistent with the run_mlm.py we need to add `mode`.
    data_collator = lambda mode: DataCollatorForCausalMultiTurnSeq2Seq(  # noqa: E731
        tokenizer,
        # Note that if you do not use `pad_to_max_length`, this becomes very slow on multi-gpus.
        padding="max_length" if data_args.pad_to_max_length else True,
        max_length=data_args.max_seq_length,
        pad_to_multiple_of=8 if training_args.fp16 else None,
    )

    # Initialize our Trainer
    trainer = ARTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
        if (training_args.do_eval or training_args.do_predict)
        else None,
    )
    # Training
    if training_args.do_train:
        checkpoint = resolve_last_checkpoint_vs_resume_from_checkpoint(
            last_checkpoint,
            training_args.resume_from_checkpoint,
        )
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples
            if data_args.max_train_samples is not None
            else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.save_model()  # Saves the tokenizer too for easy upload

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
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


if __name__ == "__main__":
    main()
