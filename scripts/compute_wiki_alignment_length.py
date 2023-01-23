from datasets import DatasetDict, Dataset, load_dataset
import datasets
from transformers import AutoTokenizer
import numpy as np

dataset_folder = "/net/nfs.cirrascale/s2-research/rabeehk/simplex-diffusion/datasets/wiki_alignment/"
raw_datasets = DatasetDict()
for split in ["train", "valid", "test"]:
    dataset = load_dataset("json", data_files=f"{dataset_folder}/{split}.jsonl")["train"]
    data_split = split if split != "valid" else "dev"
    raw_datasets[data_split] = dataset

tokenizer = AutoTokenizer.from_pretrained("roberta-large", use_fast=True)
for split in ["train", "dev", "test"]:
    original_lengths = []
    simplification_lengths = []
    data = raw_datasets[split]
    for d in data:
        token_ids = tokenizer(d["src"], padding=False, truncation=False)["input_ids"]
        original_lengths.append(len(token_ids))
        token_ids = tokenizer(d["trg"], padding=False, truncation=False)["input_ids"]
        simplification_lengths.append(len(token_ids))

    print("src ", np.mean(original_lengths), " ", np.std(original_lengths), " ", np.max(original_lengths))
    print(
        "trg ",
        np.mean(simplification_lengths),
        " ",
        np.std(simplification_lengths),
        " ",
        np.max(simplification_lengths),
    )
