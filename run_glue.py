""" Finetuning the library models for sequence classification on GLUE."""

import logging
import os
import random
import sys
from dataclasses import dataclass
from transformers.trainer_callback import TrainerState
import datasets
import numpy as np
from datasets import load_dataset
import pdb
import transformers
from transformers import AutoTokenizer, HfArgumentParser, set_seed
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version
from sdlm.arguments import ModelArguments, DiffusionArguments, TrainingArguments
from sdlm.arguments import DataTrainingArguments as BaseDataTrainingArguments
from sdlm.models import RobertaDiffusionConfig, RobertaForDiffusionLM
from sdlm.schedulers import SimplexDDPMScheduler
from sdlm.data.data_utils import split_glue
from sdlm.utils import round_stsb_target, lmap
from sdlm.data.data_collator import DataCollatorForSeq2Seq
from sdlm.trainer import DiffusionTrainer
from sdlm.inference.inference_utils import process_text
from sdlm.metrics.metrics import get_glue_metrics
from sdlm.data.postprocessors import get_post_processor
from transformers.utils import WEIGHTS_NAME

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

    if training_args.checkpoint_best_model:
        # TODO: ask which one they report and use the one needed here.
        # TODO: test both simplex and logits.
        training_args.metric_for_best_model = "pred_texts_from_simplex_masked_" + task_to_metric[data_args.dataset_name]

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
    if data_args.split_glue:
        raw_datasets = split_glue(raw_datasets, data_args.dataset_name, data_args.glue_split_seed) 
    elif data_args.dataset_name == "mnli":
        raw_datasets["validation"] =  raw_datasets["validation_matched"] # mismatched is for reverse, and for normal is matched.
        raw_datasets["test"] = raw_datasets["test_matched"]
      
    # Labels
    is_regression = data_args.dataset_name == "stsb"
    config = RobertaDiffusionConfig.from_pretrained(
        model_args.model_name_or_path,
        self_condition=diffusion_args.self_condition,
        self_condition_zeros_after_softmax=diffusion_args.self_condition_zeros_after_softmax,
        deepmind_conditional=diffusion_args.deepmind_conditional,
        classifier_free_simplex_inputs=diffusion_args.classifier_free_simplex_inputs,
        classifier_free_uncond_input=diffusion_args.classifier_free_uncond_input,
        self_condition_mlp_projection=diffusion_args.self_condition_mlp_projection,
        self_condition_mix_before_weights=diffusion_args.self_condition_mix_before_weights,
        empty_token_be_mask=diffusion_args.empty_token_be_mask,
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
        targets = [str(round_stsb_target(label)) if is_regression else str(label) for label in examples["label"]]
        labels = tokenizer(text_target=targets, max_length=max_seq_length, padding=False, truncation=True)
        max_label_length = MAX_LABEL_LENGTH  # max([len(label) for label in labels["input_ids"]])

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
        result = tokenizer(*args, padding=False, max_length=max_seq_length - max_label_length, truncation=True)
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
        predict_datasets = [raw_datasets["test"]] if data_args.dataset_name != "mnli" else [raw_datasets["test_matched"]]
        if data_args.dataset_name == "mnli":
            predict_datasets.append(raw_datasets["test_mismatched"])

        if data_args.max_predict_samples is not None:
            for i in range(len(predict_datasets)):
                max_predict_samples = min(len(predict_datasets[i]), data_args.max_predict_samples)
                predict_datasets[i] = predict_datasets[i].select(range(max_predict_samples))

    # Log a few random samples from the training set:
    if training_args.do_train:
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # Get the metric function
    task_metrics = get_glue_metrics(data_args.dataset_name)

    def postprocess_text(texts):
        # TODO: maybe we need it for others as well.
        return lmap(str.strip, texts)

    # TODO: we maybe need to pad till the sentences, and then predict the tokens we need for the few ones we need.
    def compute_metrics(results):
        post_processor = get_post_processor(data_args.dataset_name)

        # TODO: we need to change the metrics here.
        keys = ["pred_texts_from_simplex_masked", "pred_texts_from_logits_masked"]
        decoded_labels = postprocess_text(process_text(results["gold_texts_masked"]))
        if post_processor is not None:
            decoded_labels = [post_processor(x) for x in decoded_labels]

        metrics = {}
        for key in keys:
            decoded_preds = postprocess_text(process_text(results[key]))
            if post_processor is not None:
                decoded_preds = [post_processor(x) for x in decoded_preds]
            key_metrics = {}
            for metric in task_metrics:
                key_metrics.update(metric(predictions=decoded_preds, targets=decoded_labels))
            if len(key_metrics) > 1:
                key_metrics["combined_score"] = np.mean(list(key_metrics.values())).item()
            key_metrics = {f"{key}_{k}": v for k, v in key_metrics.items()}
            metrics.update(key_metrics)

        return metrics

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
    # TODO: here we need to make sure we mask to the maximum number of tokens in the labels to not signal the model for the labels.

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
        compute_metrics=compute_metrics if (training_args.do_eval or training_args.do_predict) else None,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics if (training_args.do_eval or training_args.do_predict) else None,
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

    # We will load the best model here to avoid an issue when do_train is not set.
    if training_args.load_states_in_eval_from_model_path and not training_args.do_train:
        trainer.state = TrainerState.load_from_json(os.path.join(model_args.model_name_or_path, "trainer_state.json"))
        if training_args.load_best_model_at_end and trainer.state.best_model_checkpoint is not None:
            checkpoint_path = trainer.state.best_model_checkpoint
        else:
            checkpoint_path = model_args.model_name_or_path
        trainer._load_from_checkpoint(checkpoint_path)
        trainer._load_rng_state(checkpoint_path)

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.do_predict:
        logger.info("*** Test ***")
        for i, predict_dataset in enumerate(predict_datasets):
            metric_key_prefix=f"test_{i}"
            metrics = trainer.evaluate(eval_dataset=predict_dataset, metric_key_prefix=metric_key_prefix)
            max_predict_samples = (
                data_args.max_predict_samples if data_args.max_predict_samples is not None else len(predict_dataset)
            )
            metrics["test_samples"] = min(max_predict_samples, len(predict_dataset))
            trainer.log_metrics(metric_key_prefix, metrics)
            trainer.save_metrics(metric_key_prefix, metrics)


if __name__ == "__main__":
    main()
