from datasets import DatasetDict, Dataset
import datasets
from transformers import AutoTokenizer
import numpy as np

raw_datasets = datasets.DatasetDict()
for split in ["train", "dev", "test"]:
    s1s = open(f"wikilarge/s1.{split}", "r").readlines()
    s2s = open(f"wikilarge/s2.{split}", "r").readlines()
    data = [{"original": s1, "simplification": s2} for s1, s2 in zip(s1s, s2s)]
    raw_datasets[split] = Dataset.from_list(data)

tokenizer = AutoTokenizer.from_pretrained("roberta-large", use_fast=True)
for split in ["train", "dev", "test"]:
    original_lengths = []
    simplification_lengths = []
    data = raw_datasets[split]
    for d in data:
        token_ids = tokenizer(d["original"], padding=False, truncation=False)["input_ids"]
        original_lengths.append(len(token_ids))
        token_ids = tokenizer(d["simplification"], padding=False, truncation=False)["input_ids"]
        simplification_lengths.append(len(token_ids))

    print("original ", np.mean(original_lengths), " ", np.std(original_lengths), " ", np.max(original_lengths))
    print(
        "simplification ",
        np.mean(simplification_lengths),
        " ",
        np.std(simplification_lengths),
        " ",
        np.max(simplification_lengths),
    )
