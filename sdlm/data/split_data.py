from datasets import load_from_disk, DatasetDict
import pdb

tokenized_data_path = "/net/nfs.cirrascale/s2-research/rabeehk/simplex-diffusion/processed_data/openwebtext_100"
output_dir = "/net/nfs.cirrascale/s2-research/rabeehk/simplex-diffusion/processed_data/openwebtext_100_split"
seed = 42
validation_split_ratio = 0.001

tokenized_datasets = load_from_disk(tokenized_data_path)
train_testvalid = tokenized_datasets["train"].train_test_split(test_size=validation_split_ratio, shuffle=True, seed=seed)
tokenized_datasets = DatasetDict({"train": train_testvalid["train"], "validation": train_testvalid["test"]})
tokenized_datasets.save_to_disk(output_dir)
