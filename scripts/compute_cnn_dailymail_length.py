from datasets import DatasetDict, Dataset
import datasets
from transformers import AutoTokenizer
import numpy as np
from datasets import load_dataset
import pdb

raw_datasets = load_dataset("cnn_dailymail", '3.0.0')

pdb.set_trace()
tokenizer = AutoTokenizer.from_pretrained("roberta-large", use_fast=True)
for split in ["validation", "train", "test"]:
    original_lengths = []
    simplification_lengths = []
    data = raw_datasets[split]
    for d in data:
        token_ids = tokenizer(d["article"], padding=False, truncation=False)["input_ids"]
        original_lengths.append(len(token_ids))
        token_ids = tokenizer(d["highlights"], padding=False, truncation=False)["input_ids"]
        simplification_lengths.append(len(token_ids))

    print("original ", np.mean(original_lengths), " ", np.std(original_lengths), " ", np.max(original_lengths))
    print(
        "summary ",
        np.mean(simplification_lengths),
        " ",
        np.std(simplification_lengths),
        " ",
        np.max(simplification_lengths),
    )
