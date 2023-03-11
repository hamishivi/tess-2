from datasets import DatasetDict, load_from_disk

tokenized_data_path = (
    "/home/lily/jt856/documents/simplex-diffusion/processed_data/openwebtext_50"
)
output_dir = (
    "/home/lily/jt856/documents/simplex-diffusion/processed_data/openwebtext_50_split"
)
seed = 42
validation_split_ratio = 0.001

tokenized_datasets = load_from_disk(tokenized_data_path)
train_testvalid = tokenized_datasets["train"].train_test_split(
    test_size=validation_split_ratio, shuffle=True, seed=seed
)
tokenized_datasets = DatasetDict(
    {"train": train_testvalid["train"], "validation": train_testvalid["test"]}
)
tokenized_datasets.save_to_disk(output_dir)
