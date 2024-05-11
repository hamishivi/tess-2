""" Finetuning the library models for sequence classification on GLUE."""

import logging
import os
import random
import sys

import datasets
import transformers
from datasets import load_dataset
from transformers import AutoTokenizer, set_seed
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version

from .arguments import get_args
from .data.data_collator import DataCollatorForCausalLMSeq2Seq
from .data.data_utils import split_glue
from .data.postprocessors import postprocess_text_for_metric
from .data.sni.sni_collator import DataCollatorForNI
from .metrics.metrics import get_glue_metrics
from .models import load_model
from .trainer_ar import ARTrainer

# This is computed with scripts/compute_max_tokens_of_labels.py
MAX_LABEL_LENGTH = 5
check_min_version("4.25.0")

require_version("datasets>=1.8.0")

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
    "sni": ("inputs", None),
}

task_to_metric = {
    "cola": "matthews_correlation",
    "mnli": "accuracy",
    "mrpc": "combined_score",
    "qnli": "accuracy",
    "qqp": "combined_score",
    "rte": "accuracy",
    "sst2": "accuracy",
    "stsb": "combined_score",
    "wnli": "accuracy",
    "sni": "rouge",
}

logger = logging.getLogger(__name__)


def main():
    # parse args
    model_args, data_args, training_args, diffusion_args = get_args()
    assert data_args.dataset_name is not None
    data_args.dataset_name = data_args.dataset_name.lower()
    if data_args.dataset_name not in task_to_keys.keys():
        raise ValueError(
            "Unknown task, you should pick one in " + ",".join(task_to_keys.keys())
        )

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_glue", model_args, data_args)

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
    last_checkpoint = None
    if (
        os.path.isdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif (
            last_checkpoint is not None and training_args.resume_from_checkpoint is None
        ):
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

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

    # Downloading and loading a dataset from the hub.
    if data_args.dataset_name == "sni":
        raw_datasets = load_dataset(
            "sdlm/data/sni/sni_dataset.py",
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
        )
        # sni has validation / test
        raw_datasets["validation"] = raw_datasets["test"]
        # map into simple (inputs, labels) format
        # makes easy to explore few-shot formats if we want.
        collator = DataCollatorForNI(
            tokenizer,
            text_only=True,
            num_pos_examples=0,
            max_source_length=data_args.max_source_length,
            max_target_length=data_args.max_target_length,
        )
        raw_datasets = raw_datasets.map(
            collator,
            batched=False,
            num_proc=12,  # lazy hardcode
            # load_from_cache_file=False,
        )
    else:
        raw_datasets = load_dataset(
            "glue",
            data_args.dataset_name,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
        )

    # Split dataset, since test sets of GLUE do not have the labels.
    if data_args.split_glue:
        raw_datasets = split_glue(
            raw_datasets, data_args.dataset_name, data_args.glue_split_seed
        )
    elif data_args.dataset_name == "mnli":
        raw_datasets["validation"] = raw_datasets[
            "validation_matched"
        ]  # mismatched is for reverse, and for normal is matched.
        raw_datasets["test"] = raw_datasets["test_matched"]

    # shuffle our datasets with the split_seed (split glue does this but otherwise not.)
    raw_datasets = raw_datasets.shuffle(data_args.glue_split_seed)

    # load model
    tokenizer, model = load_model(
        model_args, data_args, training_args, diffusion_args, logger
    )

    # Preprocessing the raw_datasets
    sentence1_key, sentence2_key = task_to_keys[data_args.dataset_name]

    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    def preprocess_function(examples):
        # TODO: here max_length should be max_length minus length of labels.
        # TODO: this is for now, but maybe compute one max_length as a whole.
        # Tokenize the labels.
        targets = [str(label) for label in examples["label"]]
        # we have to set this, truncate.
        max_sni_lengths = 128
        labels = tokenizer(
            text_target=targets,
            max_length=max_seq_length
            if data_args.dataset_name != "sni"
            else max_sni_lengths,
            padding=False,
            truncation=True,
        )
        # sni has long responses, while glue is all classification
        max_label_length = (
            MAX_LABEL_LENGTH if data_args.dataset_name != "sni" else max_sni_lengths
        )

        args = (
            (examples[sentence1_key],)
            if sentence2_key is None
            else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(
            *args,
            padding=False,
            max_length=max_seq_length - max_label_length,
            truncation=True,
        )
        result["labels"] = labels["input_ids"]
        return result

    with training_args.main_process_first(desc="dataset map pre-processing"):
        raw_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            load_from_cache_file=not data_args.overwrite_cache,
            num_proc=data_args.preprocessing_num_workers,
            desc="Running tokenizer on dataset",
        )
    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))

    if training_args.do_eval:
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))

    if (
        training_args.do_predict
        or data_args.dataset_name is not None
        or data_args.test_file is not None
    ):
        if "test" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_datasets = (
            [raw_datasets["test"]]
            if data_args.dataset_name != "mnli"
            else [raw_datasets["test_matched"]]
        )
        if data_args.dataset_name == "mnli":
            predict_datasets.append(raw_datasets["test_mismatched"])

        if data_args.max_predict_samples is not None:
            for i in range(len(predict_datasets)):
                max_predict_samples = min(
                    len(predict_datasets[i]), data_args.max_predict_samples
                )
                predict_datasets[i] = predict_datasets[i].select(
                    range(max_predict_samples)
                )

    # Log a few random samples from the training set:
    if training_args.do_train:
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # Get the metric function
    metric = get_glue_metrics(data_args.dataset_name)[0]

    def compute_metrics(eval_preds):
        import numpy as np

        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        # Replace -100s used for padding as we can't decode them
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        # Some simple post-processing
        decoded_preds, decoded_labels = postprocess_text_for_metric(
            "rouge", decoded_preds, decoded_labels
        )
        result = metric(predictions=decoded_preds, targets=decoded_labels)
        result = {k: round(v * 100, 4) for k, v in result.items()}
        prediction_lens = [
            np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds
        ]
        result["gen_len"] = np.mean(prediction_lens)
        return result

    # Data collator. To be consistent with the run_mlm.py we need to add `mode`.
    data_collator = lambda mode: DataCollatorForCausalLMSeq2Seq(  # noqa: E731
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
        # preprocess_logits_for_metrics=preprocess_logits_for_metrics
        # if (training_args.do_eval or training_args.do_predict)
        # else None,
        # data_args=data_args,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
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

    if training_args.do_predict:
        logger.info("*** Test ***")
        for i, predict_dataset in enumerate(predict_datasets):
            metric_key_prefix = f"test_{i}"
            metrics = trainer.evaluate(
                eval_dataset=predict_dataset, metric_key_prefix=metric_key_prefix
            )
            max_predict_samples = (
                data_args.max_predict_samples
                if data_args.max_predict_samples is not None
                else len(predict_dataset)
            )
            metrics["test_samples"] = min(max_predict_samples, len(predict_dataset))
            trainer.log_metrics(metric_key_prefix, metrics)
            trainer.save_metrics(metric_key_prefix, metrics)


if __name__ == "__main__":
    main()
