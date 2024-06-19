import os

os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from collections import defaultdict  # noqa: E402

from datasets import load_dataset  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

ds = load_dataset("sdlm/data/dolma/dolma_dataset.py", streaming=True)

text_column_name = "text"
ds = ds.select_columns([text_column_name, "source"])
ds["train"] = ds["train"].shuffle(seed=42, buffer_size=10_000)


tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
tokenizer.add_special_tokens({"pad_token": "[PAD]"})


def tokenize_function(examples):
    """
    from sdlm/data/data_utils.py (`tokenize_data_new`)
    """
    # Remove empty lines
    examples[text_column_name] = [
        line
        for line in examples[text_column_name]
        if len(line) > 0 and not line.isspace()
    ]
    return tokenizer(
        examples[text_column_name],
        # hard coded
        padding="max_length",
        truncation=True,
        # hard coded
        max_length=512,
        return_special_tokens_mask=True,
    )


tokenized_datasets = ds.map(
    tokenize_function,
    batched=True,
    remove_columns=[text_column_name],
)


def simple_collate_fn(xs):
    """simple collate fn that collects key-values from dict"""
    result = defaultdict(list)
    for x in xs:
        for key, value in x.items():
            result[key].append(value)
    return result


def source_collat_fn(xs):
    result = simple_collate_fn(xs)
    return result["source"]


def tokenize_collate_fn(xs):
    result = simple_collate_fn(xs)
    return tokenize_function(result)


# returns source information
source_dataloader = DataLoader(
    ds["train"],
    batch_size=8,
    num_workers=64,
    collate_fn=source_collat_fn,
    persistent_workers=True,
    prefetch_factor=2,
)

# returns tokens; current method via ds.map (very slow)
# also freezes if num_workers is too big ( > 1)
token_dataloader_v1 = DataLoader(
    tokenized_datasets["train"],
    batch_size=8,
    num_workers=1,
    persistent_workers=True,
    prefetch_factor=2,
)

# returns tokens; grab text and tokenize in collate_fn on the fly
token_dataloader_v2 = DataLoader(
    ds["train"],
    batch_size=8,
    num_workers=64,
    collate_fn=tokenize_collate_fn,
    persistent_workers=True,
    prefetch_factor=4,
)

# change params to test
stop_iter = 1_000
dataloader_to_test = token_dataloader_v1

for i, x in enumerate(dataloader_to_test):
    if i == stop_iter:
        break
    # just check iteration speed
    print(i)
    # check content (for source)
    # print(i, x)
