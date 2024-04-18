"""
Fine-tuning the library models for sequence to sequence.
Specifically for instruction tuning.
Runs alpacaEval as an intermediate set.
"""

import logging
import os
import sys
from dataclasses import dataclass
from typing import Optional, Union

# import alpaca_eval
import datasets
import torch
import transformers
from datasets import load_dataset
from transformers import PreTrainedTokenizerBase, set_seed
from transformers.trainer_callback import TrainerState
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import PaddingStrategy, check_min_version
from transformers.utils.versions import require_version

from .arguments import get_args

# from .data.data_collator import DataCollatorForSeq2Seq
from .data.data_utils import load_data

# from .inference.inference_utils import process_text
from .models import load_model
from .schedulers import TokenWiseSimplexDDPMScheduler
from .trainer_diffusion import DiffusionTrainer

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.25.0")
require_version("datasets>=1.8.0")
logger = logging.getLogger(__name__)


# from the open-instruct codebase.
def encode_with_messages_format(
    example, tokenizer, max_seq_length, return_string=False
):
    """
    Here we assume each example has a 'messages' field Each message is a dict with 'role' and 'content' fields.
    We concatenate all messages with the roles as delimiters and tokenize them together.
    """
    messages = example["messages"]
    if len(messages) == 0:
        raise ValueError("messages field is empty.")

    def _concat_messages(messages):
        message_text = ""
        for message in messages:
            if message["role"] == "system":
                message_text += "<|system|>\n" + message["content"].strip() + "\n"
            elif message["role"] == "user":
                message_text += "<|user|>\n" + message["content"].strip() + "\n"
            elif message["role"] == "assistant":
                message_text += (
                    "<|assistant|>\n"
                    + message["content"].strip()
                    + tokenizer.eos_token
                    + "\n"
                )
            else:
                raise ValueError("Invalid role: {}".format(message["role"]))
        return message_text

    example_text = tokenizer.bos_token + _concat_messages(messages).strip()
    tokenized_example = tokenizer(
        example_text,
        add_special_tokens=False,
        return_tensors="pt",
        max_length=max_seq_length,
        truncation=True,
    )
    input_ids = tokenized_example.input_ids
    labels = input_ids.clone()
    if return_string:
        return example_text

    # mask the non-assistant part for avoiding loss
    for message_idx, message in enumerate(messages):
        if message["role"] != "assistant":
            if message_idx == 0:
                message_start_idx = 0
            else:
                message_start_idx = tokenizer(
                    _concat_messages(messages[:message_idx]),
                    return_tensors="pt",
                    max_length=max_seq_length,
                    truncation=True,
                ).input_ids.shape[1]
            if (
                message_idx < len(messages) - 1
                and messages[message_idx + 1]["role"] == "assistant"
            ):
                # here we also ignore the role of the assistant
                messages_so_far = (
                    _concat_messages(messages[: message_idx + 1]) + "<|assistant|>\n"
                )
            else:
                messages_so_far = _concat_messages(messages[: message_idx + 1])
            message_end_idx = tokenizer(
                messages_so_far,
                return_tensors="pt",
                max_length=max_seq_length,
                truncation=True,
                add_special_tokens=False,
            ).input_ids.shape[1]
            # we replace with pad token id,
            labels[:, message_start_idx:message_end_idx] = -100

            if message_end_idx >= max_seq_length:
                break

    attention_mask = torch.ones_like(input_ids)
    return {
        "input_ids": input_ids.flatten(),
        "labels": labels.flatten(),
        "attention_mask": attention_mask.flatten(),
    }


# custom collator for the multi-turn input format.
@dataclass
class DataCollatorForMultiTurnSeq2Seq:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):
        input_ids = [feature["input_ids"] for feature in features]
        labels = [feature["labels"] for feature in features]
        features = self.tokenizer.pad(
            {"input_ids": input_ids},
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        # pad labels out for easy mask
        label_features = self.tokenizer.pad(
            {"input_ids": labels},
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )["input_ids"]
        # true wherever we have an actual label
        masks = torch.where(label_features == -100, False, True)
        features["span_mask"] = torch.tensor(masks)
        if "attention_mask" in features:
            features.pop("attention_mask")
        return features


def main():
    # parse args
    model_args, data_args, training_args, diffusion_args = get_args()

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
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if (
        os.path.isdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif (
            last_checkpoint is not None and training_args.resume_from_checkpoint is None
        ):
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # load data
    raw_datasets = load_data(data_args, model_args)
    eval_dataset = load_dataset("tatsu-lab/alpaca_eval")["eval"]

    # load model
    tokenizer, model = load_model(
        model_args, data_args, training_args, diffusion_args, logger
    )
    tokenizer.add_eos_token = False  # since the chat template adds it

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    vocab_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > vocab_size:
        model.resize_token_embeddings(len(tokenizer))

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    if training_args.do_train:
        train_column_names = raw_datasets["train"].column_names
    # if training_args.do_eval:
    #     eval_column_names = eval_dataset.column_names

    # Temporarily set max_target_length for training.
    max_target_length = data_args.max_seq_length
    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
        with training_args.main_process_first(desc="train dataset map pre-processing"):
            # we assume the data is in the tulu format
            train_dataset = train_dataset.map(
                lambda x: encode_with_messages_format(x, tokenizer, max_target_length),
                batched=False,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
                remove_columns=train_column_names,
                desc="Running tokenizer on train dataset",
            )
            train_dataset.set_format("pt")
            train_dataset = train_dataset.filter(lambda x: (x["labels"] != -100).any())

    if training_args.do_eval:
        logger.warn(
            "Running evaluation. This calls GPT-4, so PLEASE MAKE SURE YOU ARE NOT RUNNING IT A TONNE"
        )
        max_target_length = data_args.max_seq_length
        # put the dataset into the correct format
        eval_dataset = eval_dataset.map(
            lambda x: {"messages": [{"role": "user", "content": x["instruction"]}]}
        )
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))
        with training_args.main_process_first(
            desc="validation dataset map pre-processing"
        ):
            tokenized_data = []
            for sample in eval_dataset:
                prompt = encode_with_messages_format(
                    sample, tokenizer, max_target_length, return_string=True
                )
                prompt = prompt + "\n<|assistant|>\n"
                tokenized_data.append(prompt)
            data = tokenizer(
                tokenized_data,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_target_length,
                add_special_tokens=False,
            )
            eval_dataset = datasets.Dataset.from_dict(data)
            labels = []
            for sample in eval_dataset["input_ids"]:
                labels.append(
                    [-100 if x == tokenizer.pad_token_id else x for x in sample]
                )
            eval_dataset = eval_dataset.add_column("labels", labels)

    def preprocess_logits_for_metrics(logits):
        return logits.argmax(dim=-1)

    # Data collator. To be consistent with the run_mlm.py we need to add `mode`.
    data_collator = lambda mode: DataCollatorForMultiTurnSeq2Seq(  # noqa: E731
        tokenizer,
        # Note that if you do not use `pad_to_max_length`, this becomes very slow on multi-gpus.
        padding="max_length" if data_args.pad_to_max_length else True,
        max_length=data_args.max_seq_length,
        pad_to_multiple_of=8 if training_args.fp16 else None,
    )

    noise_scheduler = TokenWiseSimplexDDPMScheduler(
        num_train_timesteps=diffusion_args.num_diffusion_steps,
        beta_schedule=diffusion_args.beta_schedule,
        simplex_value=diffusion_args.simplex_value,
        clip_sample=diffusion_args.clip_sample,
        device=training_args.device,
        # multiply_factor=diffusion_args.multiply_factor,
    )
    inference_noise_schedulers = [
        TokenWiseSimplexDDPMScheduler(
            num_train_timesteps=timesteps,
            beta_schedule=diffusion_args.beta_schedule,
            simplex_value=diffusion_args.simplex_value,
            clip_sample=diffusion_args.clip_sample,
            device=training_args.device,
            # multiply_factor=diffusion_args.multiply_factor,
        )
        for timesteps in diffusion_args.num_inference_diffusion_steps
    ]

    # Metric
    def compute_metrics(results):
        # keys = ["pred_texts_from_simplex_masked", "pred_texts_from_logits_masked"]
        metrics = {}
        # eval_data = load_dataset("tatsu-lab/alpaca_eval")["eval"]
        # for key in keys:
        #     decoded_preds = (
        #         process_text(results[key])
        #         if not data_args.skip_special_tokens
        #         else results[key]
        #     )
        # for each decoded sample, format into alpacaeval setup
        # decoded_preds = [{"output": y, 'instruction': x['instruction'], 'generator': 'tess2'} for x, y in zip(eval_data, decoded_preds)]
        # df_leaderboard, _ = alpaca_eval.evaluate(
        #     decoded_preds,
        #     is_overwrite_leaderboard=True,
        #     is_return_instead_of_print=True
        # )
        # # grab tess2 results
        # key_metrics = df_leaderboard.loc['tess2'].to_dict()
        # metrics.update(key_metrics)
        return metrics

    # Initialize our Trainer
    trainer = DiffusionTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
        if (training_args.do_eval or training_args.do_predict)
        else None,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics
        if (training_args.do_eval or training_args.do_predict)
        else None,
        noise_scheduler=noise_scheduler,
        diffusion_args=diffusion_args,
        data_args=data_args,
        inference_noise_schedulers=inference_noise_schedulers,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples
            if data_args.max_train_samples is not None
            else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # We will load the best model here to avoid an issue when do_train is not set.
    if training_args.load_states_in_eval_from_model_path and not training_args.do_train:
        trainer.state = TrainerState.load_from_json(
            os.path.join(model_args.model_name_or_path, "trainer_state.json")
        )
        if (
            training_args.load_best_model_at_end
            and trainer.state.best_model_checkpoint is not None
        ):
            checkpoint_path = trainer.state.best_model_checkpoint
        else:
            checkpoint_path = model_args.model_name_or_path
        trainer._load_from_checkpoint(checkpoint_path)
        trainer._load_rng_state(checkpoint_path)

    # Evaluation
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        max_eval_samples = (
            data_args.max_eval_samples
            if data_args.max_eval_samples is not None
            else len(eval_dataset)
        )
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
    return results


if __name__ == "__main__":
    main()
