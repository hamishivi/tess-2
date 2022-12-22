"""Evaluates the GPT-2 model results. This script runs on 1 GPU."""
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging
import sys
import os
from sdlm.arguments import DataTrainingArguments, TrainingArguments, ModelArguments, DiffusionArguments
import transformers
from transformers import HfArgumentParser, set_seed
from torch.utils.data import DataLoader
import datasets
from datasets import load_from_disk
from sdlm.data.data_utils import load_data_new, tokenize_data_new
from sdlm.data.data_collator import SpanInfillingDataCollator
from sdlm.inference.inference_utils import evaluate_generation
import pdb
import numpy as np
import itertools
import math

logger = logging.getLogger(__name__)


def prepare_inputs(inputs, device):
    return {k: v.to(device) for k, v in inputs.items()}


def main():
    parser = HfArgumentParser((DataTrainingArguments, TrainingArguments, ModelArguments, DiffusionArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        data_args, training_args, model_args, diffusion_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        data_args, training_args, model_args, diffusion_args = parser.parse_args_into_dataclasses()

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
    tokenizer.pad_token = tokenizer.eos_token
    # Huggingface requires this to be set.
    tokenizer.padding_side = "right"

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
        raw_datasets = load_data_new(data_args, model_args)
        tokenized_datasets = tokenize_data_new(data_args, tokenizer, raw_datasets, training_args)

    eval_dataset = tokenized_datasets["validation"]
    if data_args.max_eval_samples is not None:
        max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
        eval_dataset = eval_dataset.select(range(max_eval_samples))

    pad_to_multiple_of_8 = data_args.line_by_line and training_args.fp16 and not data_args.pad_to_max_length
    data_collator = SpanInfillingDataCollator(
        mode="eval",
        data_args=data_args,
        tokenizer=tokenizer,
        max_length=data_args.max_seq_length,
        seed=training_args.seed,
        pad_to_multiple_of=8 if pad_to_multiple_of_8 else None,
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
    for step, batch in enumerate(eval_dataloader):
        # De-tokenize with the roberta tokenizer.
        inputs, span_mask = batch["input_ids"], batch["span_mask"]
        all_inputs.append(inputs)
        prefixes = [input[~mask] for input, mask in zip(inputs, span_mask)]
        prefixes = roberta_tokenizer.batch_decode(prefixes, skip_special_tokens=True)
        all_prefixes.extend(prefixes)
        prefixes_inputs = tokenizer(prefixes, return_tensors="pt", padding=True)

        # Compute the average length of prefixes (it should be all around the same length).
        average_prefix_length = math.ceil(prefixes_inputs["attention_mask"].sum(1).numpy().mean())
        # TODO: myabe the more accurate one is to generate this separately for each one with its own length.
        # Mask half of the sequence.
        prefixes_inputs = prepare_inputs(prefixes_inputs, training_args.device)
        outputs = model.generate(
            input_ids=prefixes_inputs["input_ids"],
            attention_mask=prefixes_inputs["attention_mask"],
            pad_token_id=tokenizer.eos_token_id,
            max_length=data_args.max_seq_length - average_prefix_length,
            min_length=data_args.max_seq_length - average_prefix_length,
            do_sample=True,
            top_p=diffusion_args.top_p,
        )
        all_outputs.append(outputs)

    results = {}
    generated_texts = [tokenizer.batch_decode(output, skip_special_tokens=True) for output in all_outputs]
    gold_texts = [roberta_tokenizer.batch_decode(input, skip_special_tokens=True) for input in all_inputs]
    generated_texts, gold_texts = list(itertools.chain(*generated_texts)), list(itertools.chain(*gold_texts))
    total_texts = [prefix + " " + generated_text for prefix, generated_text in zip(all_prefixes, generated_texts)]
    total_texts_marked = [
        prefix + " ***" + generated_text + "***" for prefix, generated_text in zip(all_prefixes, generated_texts)
    ]
    results = {"gpt2_texts": total_texts, "gold_texts": gold_texts, "prefixes": all_prefixes}
    metrics = evaluate_generation(
        results, model, tokenizer, is_conditional_generation=True, skip_special_tokens=True, prefix_lm=data_args.prefix_lm
    )
    logger.info(metrics)
    for text in total_texts_marked:
        logger.info(text)
    os.makedirs(training_args.output_dir, exist_ok=True)
    np.save(f"{training_args.output_dir}/metrics.npy", metrics)


if __name__ == "__main__":
    main()
