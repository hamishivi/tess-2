"""Evaluates the GPT-2 model results. This script runs on 1 GPU."""
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging
import sys
import os
from sdlm.arguments import DataTrainingArguments, TrainingArguments, ModelArguments
import transformers
from transformers import HfArgumentParser, set_seed
from torch.utils.data import DataLoader
import datasets
from datasets import load_from_disk
from sdlm.data.data_utils import load_data, tokenize_data_new
from sdlm.data.data_collator import SpanInfillingDataCollator
from sdlm.inference.inference_utils import evaluate_generation
import pdb
import json
import torch

logger = logging.getLogger(__name__)


def prepare_inputs(inputs, device):
    return {k: v.to(device) for k, v in inputs.items()}


def main():
    parser = HfArgumentParser((DataTrainingArguments, TrainingArguments, ModelArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        data_args, training_args, model_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        data_args, training_args, model_args = parser.parse_args_into_dataclasses()

    os.makedirs(training_args.output_dir, exist_ok=True)

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
    # Set the verbosity to info of the Transformers logger (on main process only):
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)

    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)
    roberta_tokenizer = AutoTokenizer.from_pretrained("roberta-large", **tokenizer_kwargs)
    # Running the script requires the tokenizer to have the pad token and since gpt2 tokenizer
    # does not have it, we add the pad_token here. Also, during the generation, they use the
    # eos_token_id as the pad_token_id.
    # We need to modify the bos/eos tokens to help with process function.
    tokenizer.bos_token = roberta_tokenizer.bos_token
    tokenizer.eos_token = roberta_tokenizer.eos_token
    tokenizer.pad_token = tokenizer.eos_token
    # Huggingface requires this to be set.
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    ).to(training_args.device)
    model.eval()

    if data_args.tokenized_data_path:
        tokenized_datasets = load_from_disk(data_args.tokenized_data_path)
    else:
        raw_datasets = load_data(data_args, model_args)
        tokenized_datasets = tokenize_data_new(data_args, roberta_tokenizer, raw_datasets, training_args)

    eval_dataset = tokenized_datasets["validation"]
    if data_args.max_eval_samples is not None:
        max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
        eval_dataset = eval_dataset.select(range(max_eval_samples))

    pad_to_multiple_of_8 = data_args.line_by_line and training_args.fp16 and not data_args.pad_to_max_length
    data_collator = SpanInfillingDataCollator(
        mode="eval",
        data_args=data_args,
        tokenizer=roberta_tokenizer,
        max_length=data_args.max_seq_length,
        seed=training_args.seed,
        pad_to_multiple_of=8 if pad_to_multiple_of_8 else None,
        eval_context_size=data_args.eval_context_size,
    )

    # Creates the data_loader.
    eval_dataloader = DataLoader(
        eval_dataset,
        collate_fn=data_collator,
        batch_size=training_args.per_device_eval_batch_size,
    )
    all_outputs = []
    all_inputs = []
    all_prefixes = []
    all_masks = []
    with torch.no_grad():
        for i, batch in enumerate(eval_dataloader):
            # De-tokenize with the roberta tokenizer.
            inputs, span_mask = batch["input_ids"], batch["span_mask"]
            if data_args.truncation_length > 0:
                inputs = inputs[:, : -data_args.truncation_length]
                span_mask = span_mask[:, : -data_args.truncation_length]
                max_seq_length = data_args.max_seq_length - data_args.truncation_length
                assert data_args.eval_context_size < max_seq_length
            all_masks.extend(span_mask)
            all_inputs.extend(inputs)
            prefixes_tokens = [input[~mask] for input, mask in zip(inputs, span_mask)]
            prefixes = roberta_tokenizer.batch_decode(prefixes_tokens, skip_special_tokens=True)
            all_prefixes.extend(prefixes)
            # Note that output also include the prefix and we need to remove it here.
            gold_texts = [input[len(prefix) :] for input, prefix in zip(inputs, prefixes_tokens)]
            gold_texts = [tokens.cpu().numpy().tolist() for tokens in gold_texts]
            all_outputs.extend(gold_texts)

    results = {}
    gold_texts = [
        roberta_tokenizer.decode(input[mask], skip_special_tokens=True) for input, mask in zip(all_inputs, all_masks)
    ]
    results = {
        "generated_texts_masked": gold_texts,
        "gold_texts_masked": gold_texts,
        "generated_texts_masked_tokens": all_outputs,
        "prefixes": all_prefixes,
    }
    # Saves the generated results.
    with open(f"{training_args.output_dir}/generated_results.json", "w") as f:
        json.dump(results, f)

    metrics = evaluate_generation(
        results,
        data_args,
        model,
        tokenizer,
        is_conditional_generation=True,
        prefix_lm_eval=True,
        skip_special_tokens=True,
        eval_for_all_metrics=True,
    )
    with open(f"{training_args.output_dir}/metrics.json", "w") as f:
        json.dump(metrics, f)


if __name__ == "__main__":
    main()
