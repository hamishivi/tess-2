from datasets import load_from_disk, DatasetDict
import pdb

tokenized_data_path = "/net/nfs.cirrascale/s2-research/rabeehk/simplex-diffusion/processed_data/openwebtext_256_split"
output_dir = "/net/nfs.cirrascale/s2-research/rabeehk/simplex-diffusion/processed_data/openwebtext_256_split_gpt_eval"
seed = 42
tokenized_datasets = load_from_disk(tokenized_data_path)
validation_split_ratio = 0.1414827391058291
train_testvalid = tokenized_datasets["validation"].train_test_split(
    test_size=validation_split_ratio, shuffle=True, seed=seed
)
tokenized_datasets = DatasetDict({"train": tokenized_datasets["train"], "validation": train_testvalid["test"]})
tokenized_datasets.save_to_disk(output_dir)
