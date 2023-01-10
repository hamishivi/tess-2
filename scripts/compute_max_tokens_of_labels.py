# Computes maximum numeber of tokens in the glue labels.
# from datasets import load_dataset
import pdb
from transformers import AutoTokenizer
import numpy as np

GLUE = ["mrpc", "qnli", "qqp", "rte", "sst2", "stsb", "wnli", "cola", "mnli"]
labels = {
    "cola": ["0", "1"],
    "mnli": ["0", "1", "2"],
    "mrpc": ["0", "1"],
    "rte": ["0", "1"],
    "sst2": ["0", "1"],
    "wnli": ["0", "1"],
    "stsb": [str(np.round(label, decimals=1)) for label in np.arange(0, 5.2, 0.2)],
    "qqp": ["0", "1"],
    "qnli": ["0", "1"],
}
total_max_label_length = -1
max_seq_length = 128
tokenizer_name = "roberta-large"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)

for data in GLUE:
    is_regression = data == "stsb"
    # raw_datasets = load_dataset("glue", data)
    targets = labels[data]
    targets_tokens = tokenizer(text_target=targets, max_length=max_seq_length, padding=False, truncation=True)
    max_label_length = max([len(tokens) for tokens in targets_tokens["input_ids"]])
    total_max_label_length = max(total_max_label_length, max_label_length)

print(total_max_label_length)
