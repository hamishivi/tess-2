"""
Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...) on a text file or a dataset.
"""

import logging
import os
import sys

import datasets
import evaluate
import transformers
from transformers import set_seed
from transformers.trainer_callback import TrainerState
from transformers.trainer_utils import get_last_checkpoint

from .arguments import get_args
from .data.data_collator import DataCollatorForLlamaSeq2Seq
from .data.data_utils import load_data
from .data.postprocessors import postprocess_text_for_metric
from .inference.inference_utils import process_text
from .models.utils import load_model
from .trainer_ar import ARTrainer

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
# check_min_version("4.40.0.dev0")
# require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")
logger = logging.getLogger(__name__)


summarization_name_mapping = {
    "amazon_reviews_multi": ("review_body", "review_title"),
    "big_patent": ("description", "abstract"),
    "cnn_dailymail": ("article", "highlights"),
    "orange_sum": ("text", "summary"),
    "pn_summary": ("article", "summary"),
    "psc": ("extract_text", "summary_text"),
    "samsum": ("dialogue", "summary"),
    "thaisum": ("body", "summary"),
    "xglue": ("news_body", "news_title"),
    "xsum": ("document", "summary"),
    "wiki_summary": ("article", "highlights"),
    "multi_news": ("document", "summary"),
}


def main():
    model_args, data_args, training_args, diffusion_args = get_args()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
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

    # load data
    raw_datasets = load_data(data_args, model_args)

    # load model
    tokenizer, model = load_model(
        model_args, data_args, training_args, diffusion_args, logger
    )

    total_seq2seq_length = data_args.max_source_length + data_args.max_target_length
    if (
        hasattr(model.config, "max_position_embeddings")
        and model.config.max_position_embeddings < total_seq2seq_length
    ):
        if model_args.resize_position_embeddings is None:
            logger.warning(
                "Increasing the model's number of position embedding vectors from"
                f" {model.config.max_position_embeddings} to {total_seq2seq_length}."
            )
            # position_ids starts from `padding_idx + 1` (padding_index=1) and we therefore requires
            # 2 more position embeddings.
            model.resize_position_embeddings(
                total_seq2seq_length + 2,
                with_alternatation=model_args.resize_position_embeddings_alternatively,
            )
        elif model_args.resize_position_embeddings:
            model.resize_position_embeddings(
                total_seq2seq_length + 2,
                with_alternatation=model_args.resize_position_embeddings_alternatively,
            )
        else:
            raise ValueError(
                f"`max_source_length`+`max_target_length` is set to {total_seq2seq_length}, but the model only has"
                f" {model.config.max_position_embeddings} position encodings. Consider either reducing"
                f" `max_source_length`+`max_target_length` to {model.config.max_position_embeddings} or to automatically resize the"
                " model's position encodings by passing `--resize_position_embeddings`."
            )

    # Preprocessing the datasets.
    if training_args.do_train:
        column_names = raw_datasets["train"].column_names
    elif training_args.do_eval:
        column_names = raw_datasets["validation"].column_names
    elif training_args.do_predict:
        column_names = raw_datasets["test"].column_names
    else:
        logger.info(
            "There is nothing to do. Please pass `do_train`, `do_eval` and/or `do_predict`."
        )
        return

    # Get the column names for input/target.
    dataset_columns = summarization_name_mapping.get(data_args.dataset_name, None)
    assert dataset_columns is not None, "You need to provide the columns names."
    text_column, summary_column = dataset_columns[0], dataset_columns[1]

    # Temporarily set max_target_length for training.
    max_target_length = data_args.max_target_length

    def preprocess_function(examples):
        # remove pairs where at least one record is None

        inputs, targets = [], []
        for i in range(len(examples[text_column])):
            if examples[text_column][i] and examples[summary_column][i]:
                inputs.append(examples[text_column][i])
                targets.append(examples[summary_column][i])
        # TODO: we need to process first the target, then cut the inputs to the max_length-target length to use the
        # maximum number of tokens.
        model_inputs = tokenizer(
            inputs,
            max_length=data_args.max_source_length,
            padding=False,
            truncation=True,
        )
        # Tokenize targets with the `text_target` keyword argument
        labels = tokenizer(
            text_target=targets,
            max_length=max_target_length,
            padding=False,
            truncation=True,
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
        with training_args.main_process_first(desc="train dataset map pre-processing"):
            train_dataset = train_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on train dataset",
            )

    if training_args.do_eval:
        max_target_length = data_args.val_max_target_length
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))
        with training_args.main_process_first(
            desc="validation dataset map pre-processing"
        ):
            eval_dataset = eval_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on validation dataset",
            )

    def preprocess_logits_for_metrics(logits):
        return logits.argmax(dim=-1)

    if training_args.do_predict:
        max_target_length = data_args.val_max_target_length
        if "test" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        test_dataset = raw_datasets["test"]
        if data_args.max_predict_samples is not None:
            max_predict_samples = min(len(test_dataset), data_args.max_predict_samples)
            test_dataset = test_dataset.select(range(max_predict_samples))
        with training_args.main_process_first(
            desc="prediction dataset map pre-processing"
        ):
            test_dataset = test_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on prediction dataset",
            )

    # Data collator. To be consistent with the run_mlm.py we need to add `mode`.
    data_collator = lambda mode: DataCollatorForLlamaSeq2Seq(  # noqa: E731
        tokenizer,
        # Note that if you do not use `pad_to_max_length`, this becomes very slow on multi-gpus.
        padding="max_length" if data_args.pad_to_max_length else True,
        max_length=data_args.max_seq_length,
        pad_to_multiple_of=8 if training_args.fp16 else None,
    )

    # Metric
    metric = evaluate.load("rouge")

    def compute_metrics_original(results):
        keys = ["pred_texts_from_simplex_masked", "pred_texts_from_logits_masked"]
        metrics = {}
        for key in keys:
            decoded_preds = (
                process_text(results[key])
                if not data_args.skip_special_tokens
                else results[key]
            )
            # Note that since decoded_labels is getting updated after post-process, we
            # need to compute it here for each key.
            decoded_labels = (
                process_text(results["gold_texts_masked"])
                if not data_args.skip_special_tokens
                else results["gold_texts_masked"]
            )
            decoded_preds, decoded_labels = postprocess_text_for_metric(
                "rouge", decoded_preds, decoded_labels
            )
            key_metrics = metric.compute(
                predictions=decoded_preds, references=decoded_labels, use_stemmer=True
            )
            key_metrics = {k: round(v * 100, 4) for k, v in key_metrics.items()}
            key_metrics = {f"{key}_{k}": v for k, v in key_metrics.items()}
            metrics.update(key_metrics)
        return metrics

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
        result = metric.compute(
            predictions=decoded_preds, references=decoded_labels, use_stemmer=True
        )
        result = {k: round(v * 100, 4) for k, v in result.items()}
        prediction_lens = [
            np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds
        ]
        result["gen_len"] = np.mean(prediction_lens)
        return result

    assert training_args.do_eval or training_args.do_predict

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
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()

        max_eval_samples = (
            data_args.max_eval_samples
            if data_args.max_eval_samples is not None
            else len(eval_dataset)
        )
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
        # try:
        #     perplexity = math.exp(metrics["eval_loss"])
        # except OverflowError:
        #     perplexity = float("inf")
        # metrics["perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.do_predict:
        logger.info("*** Test ***")
        metrics = trainer.evaluate(test_dataset, metric_key_prefix="test")
        max_predict_samples = (
            data_args.max_predict_samples
            if data_args.max_predict_samples is not None
            else len(test_dataset)
        )
        metrics["test_samples"] = min(max_predict_samples, len(test_dataset))
        trainer.log_metrics("test", metrics)
        trainer.save_metrics("test", metrics)


if __name__ == "__main__":
    main()
