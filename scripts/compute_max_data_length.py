# Computes the maximum length in a dataset.
# Usage: python scripts/compute_max_data_length.py --dataset_name xsum --dataset_config "3.0.0" --model_name_or_path "roberta-large"

from sdlm.data.data_utils import load_data
from transformers import AutoTokenizer, HfArgumentParser
from sdlm.arguments import ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments, DiffusionArguments
import sys
import pdb
import logging
import os
import numpy as np

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


def compute_max_length(dataset, prefix):
    data_lengths = []
    target_lengths = []
    # Compute the length.
    for x in dataset:
        data_length = len(x["input_ids"])
        target_length = len(x["labels"])
        data_lengths.append(data_length)
        target_lengths.append(target_length)

    np.save(prefix + "_target_lengths.npy", target_lengths)
    np.save(prefix + "_data_lengths.npy", data_lengths)
    print("Target length ", np.mean(target_lengths), np.std(target_lengths), np.max(target_lengths), np.min(target_lengths))
    print("Data length ", np.mean(data_lengths), np.std(data_lengths), np.max(data_lengths), np.min(data_lengths))


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments, DiffusionArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args, _ = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args, _ = parser.parse_args_into_dataclasses()

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
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
        model_inputs = tokenizer(inputs, max_length=data_args.max_source_length, padding=False, truncation=True)
        # Tokenize targets with the `text_target` keyword argument
        labels = tokenizer(text_target=targets, max_length=max_target_length, padding=False, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    raw_datasets = load_data(data_args, model_args)
    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    column_names = raw_datasets["train"].column_names

    # Get the column names for input/target.
    dataset_columns = summarization_name_mapping.get(data_args.dataset_name, None)
    assert dataset_columns is not None, "You need to provide the columns names."
    text_column, summary_column = dataset_columns[0], dataset_columns[1]

    train_dataset = raw_datasets["train"]
    validation_dataset = raw_datasets["validation"]
    test_dataset = raw_datasets["test"]
    with training_args.main_process_first(desc="train dataset map pre-processing"):
        train_dataset = train_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on train dataset",
        )
        validation_dataset = validation_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on train dataset",
        )
        test_dataset = test_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on train dataset",
        )

    compute_max_length(train_dataset, prefix="train")
    compute_max_length(validation_dataset, prefix="validation")
    compute_max_length(test_dataset, prefix="test")
    # Max data_length across train/validation/test is 1024 and max target_length is 120.


if __name__ == "__main__":
    main()
