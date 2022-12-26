""" Finetuning the library models for sequence classification on GLUE."""

import logging
import os
import random
import sys
from dataclasses import dataclass

import datasets
import numpy as np
from datasets import load_dataset
import pdb
import evaluate
import transformers
from transformers import (
    AutoTokenizer,
    EvalPrediction,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version
from sdlm.arguments import ModelArguments, DiffusionArguments
from sdlm.arguments import DataTrainingArguments as BaseDataTrainingArguments
from sdlm.models import RobertaDiffusionConfig, RobertaForDiffusionLM
from sdlm.schedulers import SimplexDDPMScheduler
from sdlm.data.data_utils import split_glue
from sdlm.utils import round_stsb_target
from sdlm.data.data_collator import DataCollatorForSeq2Seq
from sdlm.trainer import DiffusionTrainer

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
}


logger = logging.getLogger(__name__)


@dataclass
class DataTrainingArguments(BaseDataTrainingArguments):
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    def __post_init__(self):
        assert self.dataset_name is not None
        self.dataset_name = self.dataset_name.lower()
        if self.dataset_name not in task_to_keys.keys():
            raise ValueError("Unknown task, you should pick one in " + ",".join(task_to_keys.keys()))


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments, DiffusionArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args, diffusion_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args, diffusion_args = parser.parse_args_into_dataclasses()

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
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Downloading and loading a dataset from the hub.
    raw_datasets = load_dataset(
        "glue",
        data_args.dataset_name,
        cache_dir=model_args.cache_dir,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    # Split dataset, since test sets of GLUE do not have the labels.
    raw_datasets = split_glue(raw_datasets, data_args.dataset_name, training_args.seed)

    # Labels
    is_regression = data_args.dataset_name == "stsb"
    if not is_regression:
        label_list = raw_datasets["train"].features["label"].names
        num_labels = len(label_list)
    else:
        num_labels = 1

    config = RobertaDiffusionConfig.from_pretrained(
        model_args.model_name_or_path,
        self_condition=diffusion_args.self_condition,
        self_condition_zeros_after_softmax=diffusion_args.self_condition_zeros_after_softmax,
        deepmind_conditional=diffusion_args.deepmind_conditional,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    if model_args.model_name_or_path:
        model = RobertaForDiffusionLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    else:
        logger.info("Training new model from scratch")
        model = RobertaForDiffusionLM.from_config(config)

    # Preprocessing the raw_datasets
    sentence1_key, sentence2_key = task_to_keys[data_args.dataset_name]

    padding = "max_length" if data_args.pad_to_max_length else False

    """
    # Some models have set the order of the labels to use, so let's make sure we do use it.
    if not is_regression:
        model.config.label2id = {l: i for i, l in enumerate(label_list)}
        model.config.id2label = {id: label for label, id in config.label2id.items()}
    """
    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    def preprocess_function(examples):
        # Tokenize the texts.
        if data_args.add_t5_tags:
            sentence1_with_tag = [sentence1_key + ": " + sentence_1 for sentence_1 in examples[sentence1_key]]
            if sentence2_key is not None:
                sentence2_with_tag = [sentence2_key + ": " + sentence_2 for sentence_2 in examples[sentence2_key]]
            args = (sentence1_with_tag,) if sentence2_key is None else (sentence1_with_tag, sentence2_with_tag)
        else:
            args = (
                (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
            )
        result = tokenizer(*args, padding=False, max_length=max_seq_length, truncation=True)

        # Tokenize the labels.
        targets = [str(round_stsb_target(label)) if is_regression else str(label) for label in examples["label"]]
        if data_args.add_t5_tags:
            targets = ["label: " + label for label in targets]
        labels = tokenizer(text_target=targets, max_length=max_seq_length, padding=False, truncation=True)
        result["labels"] = labels["input_ids"]
        return result

    with training_args.main_process_first(desc="dataset map pre-processing"):
        raw_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            load_from_cache_file=not data_args.overwrite_cache,
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

        def preprocess_logits_for_metrics(logits):
            return logits.argmax(dim=-1)

    if training_args.do_predict or data_args.dataset_name is not None or data_args.test_file is not None:
        if "test" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = raw_datasets["test"]
        if data_args.max_predict_samples is not None:
            max_predict_samples = min(len(predict_dataset), data_args.max_predict_samples)
            predict_dataset = predict_dataset.select(range(max_predict_samples))

    # Log a few random samples from the training set:
    if training_args.do_train:
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # Get the metric function
    metric = evaluate.load("glue", data_args.dataset_name)

    # You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.
    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)
        result = metric.compute(predictions=preds, references=p.label_ids)
        if len(result) > 1:
            result["combined_score"] = np.mean(list(result.values())).item()
        return result

    # Data collator will default to DataCollatorWithPadding when the tokenizer is passed to Trainer, so we change it if
    # we already did the padding.
    # Data collator. To be consistent with the run_mlm.py we need to add `mode`.
    data_collator = lambda mode: DataCollatorForSeq2Seq(
        tokenizer,
        # Note that if you do not use `pad_to_max_length`, this becomes very slow on multi-gpus.
        padding="max_length" if data_args.pad_to_max_length else True,
        max_length=data_args.max_seq_length,
        pad_to_multiple_of=8 if training_args.fp16 else None,
    )
    noise_scheduler = SimplexDDPMScheduler(
        num_train_timesteps=diffusion_args.num_diffusion_steps,
        beta_schedule=diffusion_args.beta_schedule,
        simplex_value=diffusion_args.simplex_value,
        clip_sample=diffusion_args.clip_sample,
        device=training_args.device,
    )
    inference_noise_scheduler = SimplexDDPMScheduler(
        num_train_timesteps=diffusion_args.num_inference_diffusion_steps,
        beta_schedule=diffusion_args.beta_schedule,
        simplex_value=diffusion_args.simplex_value,
        clip_sample=diffusion_args.clip_sample,
        device=training_args.device,
    )

    # Initialize our Trainer
    trainer = DiffusionTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics if training_args.do_eval else None,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics if training_args.do_eval else None,
        noise_scheduler=noise_scheduler,
        diffusion_args=diffusion_args,
        data_args=data_args,
        inference_noise_scheduler=inference_noise_scheduler,
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
        max_train_samples = data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.save_model()  # Saves the tokenizer too for easy upload

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        # Loop to handle MNLI double evaluation (matched, mis-matched)
        tasks = [data_args.dataset_name]
        eval_datasets = [eval_dataset]

        for eval_dataset, task in zip(eval_datasets, tasks):
            metrics = trainer.evaluate(eval_dataset=eval_dataset)

            max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
            metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

            trainer.log_metrics("eval", metrics)
            trainer.save_metrics("eval", metrics)


if __name__ == "__main__":
    main()
