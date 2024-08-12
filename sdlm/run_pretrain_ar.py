# run_clm.py
import logging
import os
import sys

import datasets
import transformers
from transformers import (
    Trainer,
    default_data_collator,
    is_torch_tpu_available,
    set_seed,
)
from transformers.trainer_callback import TrainerState
from transformers.utils import check_min_version
from transformers.utils.versions import require_version

from sdlm.run_pretrain import filter_by_length

from .arguments import get_args
from .data.data_utils import load_data, tokenize_data_new
from .models import load_model
from .utils import (
    get_last_checkpoint_with_beaker_preemption,
    is_nfs_available,
    is_weka_available,
    resolve_last_checkpoint_vs_resume_from_checkpoint,
    set_hf_home,
    set_pretraining_dataset,
)

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.25.0")

require_version(
    "datasets>=2.0.0",
    "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt",
)

logger = logging.getLogger(__name__)

# set environment variables
set_hf_home()
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def main():
    # parse args
    model_args, data_args, training_args, diffusion_args = get_args()
    set_pretraining_dataset(data_args)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = get_last_checkpoint_with_beaker_preemption(training_args)

    # load model
    tokenizer, model = load_model(
        model_args, data_args, training_args, diffusion_args, logger
    )
    assert model.config.pad_token_id is not None

    # Set seed before initializing model.
    set_seed(training_args.seed)

    if training_args.do_train:
        raw_datasets = load_data(data_args, model_args)
        train_dataset = tokenize_data_new(
            data_args, tokenizer, raw_datasets, training_args
        )["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
        if data_args.min_train_seq_length != 0:
            train_dataset = train_dataset.filter(
                filter_by_length(
                    data_args.min_train_seq_length, model.config.pad_token_id
                )
            )
        if data_args.shuffle and data_args.streaming:
            train_dataset = train_dataset.shuffle(
                seed=training_args.seed, buffer_size=10_000
            )
        elif data_args.shuffle:
            train_dataset = train_dataset.shuffle(seed=training_args.seed)

        # NOTE: modifications for clm
        train_dataset = train_dataset.map(
            lambda x: {**x, "labels": x["input_ids"]},
            remove_columns=["special_tokens_mask"],
        )

    if training_args.do_eval:
        # default to c4
        if is_weka_available():
            data_file_path = "/data/input/jaket/c4_subset"
        elif is_nfs_available():
            data_file_path = (
                "/net/nfs.cirrascale/allennlp/jaket/simplex-diffusion/c4_subset"
            )
        else:
            # yale
            data_file_path = "/home/jt856/documents/simplex-diffusion/raw/c4_subset"
        c4_raw_dataset = datasets.IterableDatasetDict(
            {
                "validation": datasets.load_dataset(
                    "json",
                    data_files=os.path.join(
                        data_file_path, "c4-validation.00000-of-00008.json"
                    ),
                )["train"]
            }
        )
        c4_tokenized_datasets = tokenize_data_new(
            data_args, tokenizer, c4_raw_dataset, training_args
        )
        eval_dataset = c4_tokenized_datasets["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))
        if data_args.min_eval_seq_length != 0:
            eval_dataset = eval_dataset.filter(
                filter_by_length(
                    data_args.min_eval_seq_length, model.config.pad_token_id
                ),
                num_proc=data_args.preprocessing_num_workers,
            )
        # NOTE: modifications for clm
        eval_dataset = eval_dataset.map(
            lambda x: {**x, "labels": x["input_ids"]},
            remove_columns=["special_tokens_mask"],
        )

        def preprocess_logits_for_metrics(logits, labels):
            if isinstance(logits, tuple):
                # Depending on the model and config, logits may contain extra tensors,
                # like past_key_values, but logits always come first
                logits = logits[0]
            return logits.argmax(dim=-1)

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        # Data collator will default to DataCollatorWithPadding, so we change it.
        data_collator=default_data_collator,
        compute_metrics=None,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics
        if training_args.do_eval and not is_torch_tpu_available()
        else None,
    )

    # Training
    if training_args.do_train:
        checkpoint = resolve_last_checkpoint_vs_resume_from_checkpoint(
            last_checkpoint,
            training_args.resume_from_checkpoint,
        )
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload
        metrics = train_result.metrics

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        if training_args.load_states_in_eval_from_model_path:
            trainer._load_from_checkpoint(model_args.model_name_or_path)
            trainer.state = TrainerState.load_from_json(
                os.path.join(model_args.model_name_or_path, "trainer_state.json")
            )
            trainer._load_rng_state(model_args.model_name_or_path)

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


if __name__ == "__main__":
    main()
