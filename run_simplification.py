"""
Fine-tuning the library models for sequence to sequence.
"""

import logging
import os
import sys
import json 
import datasets
import nltk
import numpy as np 
from datasets import DatasetDict, Dataset, load_dataset
from transformers.trainer_callback import TrainerState
import evaluate
import transformers
from filelock import FileLock
from transformers import AutoTokenizer, HfArgumentParser, set_seed
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, is_offline_mode, send_example_telemetry
from transformers.utils.versions import require_version
from sdlm.arguments import ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments, DiffusionArguments
from sdlm.models import RobertaDiffusionConfig, RobertaForDiffusionLM
from sdlm.schedulers import SimplexDDPMScheduler
import pdb
from sdlm.trainer import DiffusionTrainer
from sdlm.data.data_collator import DataCollatorForSeq2Seq
from sdlm.inference.inference_utils import process_text
from sdlm.metrics.metrics import distinct_n_grams
from sdlm.data.postprocessors import postprocess_text_for_metric


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.25.0")

require_version("datasets>=1.8.0")

logger = logging.getLogger(__name__)

try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    if is_offline_mode():
        raise LookupError("Offline mode: run this script without TRANSFORMERS_OFFLINE first to download nltk data files")
    with FileLock(".lock") as lock:
        nltk.download("punkt", quiet=True)

def read_wikilarge(data_args):
    raw_datasets = DatasetDict()
    for split in ["train", "dev", "test"]:
        # TODO: change to f"{data_args.dataset_folder}/"
        s1s = open(f"wikilarge/s1.{split}", "r").readlines() 
        s2s = open(f"wikilarge/s2.{split}", "r").readlines()
        data = [{"original": s1, "simplification": s2} for s1, s2 in zip(s1s, s2s)]
        raw_datasets[split] = Dataset.from_list(data)
    return raw_datasets 

def read_diffuseq_datasets(data_args):
    raw_datasets = DatasetDict()
    for split in ["train", "valid", "test"]:
        dataset = load_dataset("json", data_files=f"{data_args.dataset_folder}/{split}.jsonl")["train"]
        data_split = split if split != "valid" else "dev"
        raw_datasets[data_split] = dataset
    return raw_datasets

simplification_name_mapping = {
    "wikilarge": ("original", "simplification"),
    "wiki_alignment": ("src", "trg"),
    "qqp": ("src", "trg"),
    "qg": ("src", "trg"),
    "cc": ("src", "trg")
}


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments, DiffusionArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args, diffusion_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args, diffusion_args = parser.parse_args_into_dataclasses()

    assert data_args.max_target_length + data_args.max_source_length <= data_args.max_seq_length

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_summarization", model_args, data_args)

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

    if data_args.dataset_name == "wikilarge":
        raw_datasets = read_wikilarge(data_args)
    elif data_args.dataset_name in ["wiki_alignment", "qqp", "qg", "cc"]:
        raw_datasets = read_diffuseq_datasets(data_args)
    train_dataset = raw_datasets["train"]
    eval_dataset = raw_datasets["dev"]
    test_dataset = raw_datasets["test"]
    column_names = train_dataset.column_names
    config = RobertaDiffusionConfig.from_pretrained(
        model_args.model_name_or_path,
        self_condition=diffusion_args.self_condition,
        self_condition_zeros_after_softmax=diffusion_args.self_condition_zeros_after_softmax,
        deepmind_conditional=diffusion_args.deepmind_conditional,
        classifier_free_simplex_inputs=diffusion_args.classifier_free_simplex_inputs,
        classifier_free_uncond_input=diffusion_args.classifier_free_uncond_input,
        self_condition_mlp_projection=diffusion_args.self_condition_mlp_projection,
        self_condition_mix_before_weights=diffusion_args.self_condition_mix_before_weights,
        self_condition_mix_logits_before_weights=diffusion_args.self_condition_mix_logits_before_weights,
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

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    vocab_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > vocab_size:
        model.resize_token_embeddings(len(tokenizer))

    total_seq2seq_length = data_args.max_source_length + data_args.max_target_length
    if hasattr(model.config, "max_position_embeddings") and model.config.max_position_embeddings < total_seq2seq_length:
        if model_args.resize_position_embeddings is None:
            logger.warning(
                "Increasing the model's number of position embedding vectors from"
                f" {model.config.max_position_embeddings} to {total_seq2seq_length}."
            )
            # position_ids starts from `padding_idx + 1` (padding_index=1) and we therefore requires
            # 2 more position embeddings.
            model.resize_position_embeddings(total_seq2seq_length + 2)
        elif model_args.resize_position_embeddings:
            model.resize_position_embeddings(total_seq2seq_length + 2)
        else:
            raise ValueError(
                f"`max_source_length`+`max_target_length` is set to {total_seq2seq_length}, but the model only has"
                f" {model.config.max_position_embeddings} position encodings. Consider either reducing"
                f" `max_source_length`+`max_target_length` to {model.config.max_position_embeddings} or to automatically resize the"
                " model's position encodings by passing `--resize_position_embeddings`."
            )

    # We need to tokenize inputs and targets.
    # Get the column names for input/target.
    dataset_columns = simplification_name_mapping.get(data_args.dataset_name, None)
    assert dataset_columns is not None, "You need to provide the columns names."
    text_column, simplification_column = dataset_columns[0], dataset_columns[1]

    # Temporarily set max_target_length for training.
    max_target_length = data_args.max_target_length

    """
    if training_args.label_smoothing_factor > 0 and not hasattr(model, "prepare_decoder_input_ids_from_labels"):
        logger.warning(
            "label_smoothing is enabled but the `prepare_decoder_input_ids_from_labels` method is not defined for"
            f"`{model.__class__.__name__}`. This will lead to loss being calculated twice and will take up more memory"
        )
    """

    def preprocess_function(examples):
        # remove pairs where at least one record is None

        inputs, targets = [], []
        for i in range(len(examples[text_column])):
            if examples[text_column][i] and examples[simplification_column][i]:
                inputs.append(examples[text_column][i])
                targets.append(examples[simplification_column][i])
        # TODO: we need to process first the target, then cut the inputs to the max_length-target length to use the
        # maximum number of tokens.
        model_inputs = tokenizer(inputs, max_length=data_args.max_source_length, padding=False, truncation=True)
        # Tokenize targets with the `text_target` keyword argument
        labels = tokenizer(text_target=targets, max_length=max_target_length, padding=False, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    if training_args.do_train:
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
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))
        with training_args.main_process_first(desc="validation dataset map pre-processing"):
            eval_dataset = eval_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on validation dataset",
            )

    if training_args.do_predict:
        max_target_length = data_args.val_max_target_length
        test_dataset = raw_datasets["test"]
        if data_args.max_predict_samples is not None:
            max_predict_samples = min(len(test_dataset), data_args.max_predict_samples)
            test_dataset = test_dataset.select(range(max_predict_samples))
        with training_args.main_process_first(desc="prediction dataset map pre-processing"):
            test_dataset = test_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on prediction dataset",
            )

    def preprocess_logits_for_metrics(logits):
        return logits.argmax(dim=-1)

    # TODO: we may want to add predict back.

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

    # Metric
    eval_metrics = {
        "sari": evaluate.load("sari"),
        "bleu": evaluate.load("bleu"),
        "bertscore": evaluate.load("bertscore"),
        "bertscore_them": evaluate.load("bertscore"),
        "rouge": evaluate.load("rouge"),
        "dist":  distinct_n_grams
    }

    def compute_metrics(results):
        keys = ["pred_texts_from_simplex_masked", "pred_texts_from_logits_masked"]
        metrics = {}
        for key in keys:
            decoded_preds_original = process_text(results[key]) if not data_args.skip_special_tokens else results[key]
            decoded_labels_original = process_text(results["gold_texts_masked"]) if not data_args.skip_special_tokens else results["gold_texts_masked"]
            sources = results["prefixes"]
            for metric_name, metric in eval_metrics.items():
                scale = 100 if metric_name != "sari" else 1
                if metric_name == "sari":
                    decoded_preds, decoded_labels, sources = postprocess_text_for_metric("sari", decoded_preds_original, decoded_labels_original, sources)
                    decoded_labels = [[decoded_label] for decoded_label in decoded_labels]
                    key_metrics = metric.compute(sources=sources, predictions=decoded_preds, references=decoded_labels)
                elif metric_name == "bleu":
                    decoded_preds, decoded_labels = postprocess_text_for_metric("bleu", decoded_preds_original, decoded_labels_original)
                    key_metrics = {"bleu": metric.compute(predictions=decoded_preds, references=decoded_labels)["bleu"]}
                elif metric_name == "bertscore":
                    decoded_preds, decoded_labels = postprocess_text_for_metric("bertscore", decoded_preds_original, decoded_labels_original)
                    key_metrics = {"bert_score": np.mean(metric.compute(predictions=decoded_preds, references=decoded_labels,  lang="en")['f1'])}
                elif metric_name == "bertscore_them":
                    decoded_preds, decoded_labels = postprocess_text_for_metric("bertscore_them", decoded_preds_original, decoded_labels_original)
                    key_metrics = {"bert_score_them": np.mean(metric.compute(predictions=decoded_preds, references=decoded_labels, model_type='microsoft/deberta-xlarge-mnli', lang="en")['f1'])}
                elif metric_name == "rouge":
                    decoded_preds, decoded_labels = postprocess_text_for_metric("rouge", decoded_preds_original, decoded_labels_original)
                    key_metrics = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
                elif metric_name == "dist":
                    decoded_preds = postprocess_text_for_metric("dist", decoded_preds_original)
                    key_metrics = metric(decoded_preds)

                key_metrics = {k: round(v*scale, 2) for k, v in key_metrics.items()}
                key_metrics = {f"{key}_{k}": v for k, v in key_metrics.items()}
                metrics.update(key_metrics)
        return metrics

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
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        max_train_samples = data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

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
    results = {}
    num_beams = data_args.num_beams if data_args.num_beams is not None else training_args.generation_num_beams
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        # TODO: num_beans should be added for ours as well.
        # metrics = trainer.evaluate(max_length=max_length, num_beams=num_beams, metric_key_prefix="eval")
        metrics = trainer.evaluate()
        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.do_predict:
        logger.info("*** Test ***")
        # TODO: num_beans should be added for ours as well.
        # metrics = trainer.evaluate(max_length=max_length, num_beams=num_beams, metric_key_prefix="eval")
        metrics = trainer.evaluate(test_dataset, metric_key_prefix="test")
        max_test_samples = data_args.max_predict_samples if data_args.max_predict_samples is not None else len(test_dataset)
        metrics["test_samples"] = min(max_test_samples, len(test_dataset))
        trainer.log_metrics(f"test", metrics)
        trainer.save_metrics(f"test", metrics)

    # TODO: we may want to add predict part back.
    return results


if __name__ == "__main__":
    main()
