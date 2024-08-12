"""Arguments used in training/inference/data processing."""
import os
import sys
from dataclasses import dataclass, field
from typing import List, Optional

from transformers import MODEL_MAPPING, HfArgumentParser, SchedulerType
from transformers import TrainingArguments as HFTrainingArguments

MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def get_args():
    parser = HfArgumentParser(
        (
            ModelArguments,
            DataTrainingArguments,
            Seq2SeqTrainingArguments,
            DiffusionArguments,
        )
    )
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args, diffusion_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        (
            model_args,
            data_args,
            training_args,
            diffusion_args,
        ) = parser.parse_args_into_dataclasses()
    return model_args, data_args, training_args, diffusion_args


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization. Don't set if you want to train a model from scratch."
            )
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={
            "help": "If training from scratch, pass a model type from the list: "
            + ", ".join(MODEL_TYPES)
        },
    )
    use_model: str = field(
        default="",
        metadata={
            "help": "Choose whether to use a cdcd or tokenwise model. Options: cdcd, tokenwise_cdcd, confidence."
        },
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override some existing default config settings when a model is trained from scratch. Example: "
                "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
            )
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )
    tokenizer_padding_side: Optional[str] = field(
        default="right",
        metadata={"help": "Tokenizer padding side"},
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from huggingface.co"
        },
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={
            "help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."
        },
    )
    model_revision: str = field(
        default="main",
        metadata={
            "help": "The specific model version to use (can be a branch name, tag name or commit id)."
        },
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    autoregressive_eval_model: str = field(
        default="EleutherAI/gpt-neo-1.3B",
        metadata={
            "help": "The autoregressive model used to measure the evaluation perplexity."
        },
    )
    resize_position_embeddings: Optional[bool] = field(
        default=None,
        metadata={
            "help": (
                "Whether to automatically resize the position embeddings if `max_source_length` exceeds "
                "the model's position embeddings."
            )
        },
    )
    resize_position_embeddings_alternatively: Optional[bool] = field(
        default=False,
        metadata={
            "help": "If set, resizes the position embedding alternatively, and copies from the original for the uncovered part."
        },
    )
    from_scratch: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Whether to train the model from scratch or not. Default to false."
        },
    )
    use_flash_attention2: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to use flash attention 2."},
    )
    is_causal: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to use causal attention (for Llama)."},
    )
    use_lora: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to use LoRA."},
    )
    lora_rank: Optional[int] = field(
        default=16,
        metadata={"help": "LoRA rank."},
    )
    lora_alpha: Optional[int] = field(
        default=32,
        metadata={"help": "LoRA alpha."},
    )
    lora_dropout: Optional[float] = field(
        default=0.1,
        metadata={"help": "LoRA dropout."},
    )
    freeze_embedding: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to freeze vocab embedding (and tied LM head)."},
    )
    freeze_model: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to freeze the entire model."},
    )

    def __post_init__(self):
        if self.config_overrides is not None and (self.model_name_or_path is not None):
            raise ValueError(
                "--config_overrides can't be used in combination with --model_name_or_path"
            )


@dataclass
class TrainingArguments(HFTrainingArguments):
    lr_scheduler_type: SchedulerType = field(
        default="linear",
        metadata={
            "help": (
                "The scheduler type to use. It can be `linear`, `cosine`,"
                "`cosine_with_restarts`, `polynomial`, `constant`, and `constant_with_warmup`"
            )
        },
    )
    output_dir: Optional[str] = field(
        default=None, metadata={"help": "Where to store the final model."}
    )
    checkpointing_steps: int = field(
        default=1000, metadata={"help": "Specifies the checkpoint step."}
    )
    resume_from_checkpoint: Optional[str] = field(
        default=None,
        metadata={"help": "If the training should continue from a checkpoint folder."},
    )
    log_generated_texts: bool = field(
        default=True, metadata={"help": "If set, logs generated texts."}
    )
    checkpoint_best_model: bool = field(
        default=False,
        metadata={
            "help": "If set, for `run_glue.py` it sets the metrics name"
            "to save the best model in each checkpoint step."
        },
    )
    eval_for_all_metrics: bool = field(
        default=False,
        metadata={"help": "If set, evaluates on all metrics in run_mlm.py"},
    )
    load_states_in_eval_from_model_path: bool = field(
        default=True,
        metadata={
            "help": "In case of only using --do_eval without --do_train, use it to load the states before eval."
            "keep this to true, it causes otherwise an issue with huggingface when doing only --do_eval."
            "This parameter when running baselines does not have any impact and is not needed."
        },
    )
    without_compute_metrics: bool = field(
        default=False,
        metadata={
            "help": "If set, does not compute the metrics. we are observing MAUVE is very slow"
            "on multi-gpu setting and we do this to compute the metrics separately."
            "If using this option, you can call `compute_mlm_metrics.py` to compute them on 1 GPU later on."
        },
    )
    compute_eval_loss_with_simplex: bool = field(
        default=False,
        metadata={
            "help": "If set, computes the evaluation loss from the simplex values."
        },
    )
    save_checkpoints_on_s3: bool = field(
        default=False,
        metadata={
            "help": "If set, instead of deleting the checkpoints when passing the limit of save checkpoints, it saves them on S3."
        },
    )
    # NOTE: change default to suppress deprecation warning
    optim: str = field(default="adamw_torch")
    # just for beaker training, to allow auto-resume easier.
    beaker: bool = field(default=False)
    mask_padding_in_loss: bool = field(
        default=False,
        metadata={"help": "Whether to mask padding token in loss computation."},
    )
    generation_config: str = field(default=None)
    timestep_embed_lr: Optional[float] = field(
        default=None, metadata={"help": "LR for timestep embedding."}
    )
    disable_timestep_embed: Optional[bool] = field(
        default=False, metadata={"help": "Whether to disable timestep embedding."}
    )


@dataclass
class Seq2SeqTrainingArguments(TrainingArguments):
    """
    Args:
        sortish_sampler (`bool`, *optional*, defaults to `False`):
            Whether to use a *sortish sampler* or not. Only possible if the underlying datasets are *Seq2SeqDataset*
            for now but will become generally available in the near future.
            It sorts the inputs according to lengths in order to minimize the padding size, with a bit of randomness
            for the training set.
        generation_max_length (`int`, *optional*):
            The `max_length` to use on each evaluation loop when `predict_with_generate=True`. Will default to the
            `max_length` value of the model configuration.
        generation_num_beams (`int`, *optional*):
            The `num_beams` to use on each evaluation loop when `predict_with_generate=True`. Will default to the
            `num_beams` value of the model configuration.
    """

    sortish_sampler: bool = field(
        default=False, metadata={"help": "Whether to use SortishSampler or not."}
    )
    generation_max_length: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "The `max_length` to use on each evaluation loop when `predict_with_generate=True`. Will default "
                "to the `max_length` value of the model configuration."
            )
        },
    )
    generation_num_beams: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "The `num_beams` to use on each evaluation loop when `predict_with_generate=True`. Will default "
                "to the `num_beams` value of the model configuration."
            )
        },
    )
    predict_with_generate: Optional[bool] = field(default=True)


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    split_glue: bool = field(
        default=False,
        metadata={
            "help": "If set to true split the glue dev/train to make the test set"
            "otherwises uses the original splits."
        },
    )
    glue_split_seed: int = field(
        default=42, metadata={"help": "Seed to split the glue data."}
    )
    is_tulu_pair: bool = field(
        default=False,
        metadata={"help": "Whether to use pair preprocessing for TULU."},
    )
    is_tulu_multiturn: bool = field(
        default=False,
        metadata={"help": "Whether to use multiturn preprocessing for TULU."},
    )
    is_tulu_sliding_window_multiturn: bool = field(
        default=False,
        metadata={
            "help": "Whether to use sliding window multiturn preprocessing for TULU."
        },
    )
    ul2_max_mask_ratio: float = field(
        default=0.5,
        metadata={"help": "UL2 variable maximum mask ratio."},
    )
    tokenized_data_path: Optional[str] = field(
        default=None, metadata={"help": "If set, reads a tokenized train data."}
    )
    dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the dataset to use (via the datasets library)."},
    )
    dataset_config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The configuration name of the dataset to use (via the datasets library)."
        },
    )
    dataset_folder: str = field(
        default=None, metadata={"help": "The dataset folder containing the dataset."}
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a text file)."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."
        },
    )
    test_file: Optional[str] = field(
        default=None, metadata={"help": "A text file containing the test data."}
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
    )
    validation_split_ratio: Optional[float] = field(
        default=0.001,
        metadata={
            "help": "The ratio(< 1.0) of the train set used as validation set in case there's no validation split."
        },
    )
    max_seq_length: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated."
            )
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    line_by_line: bool = field(
        default=False,
        metadata={
            "help": "Whether distinct lines of text in the dataset are to be handled as distinct sequences."
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to pad all samples to `max_seq_length`. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch."
            )
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    mask_ratio: float = field(
        default=0.15,
        metadata={
            "help": "Defines the ratio of mask tokens. A number between 0 and 1."
        },
    )
    mean_mask_span_length: int = field(
        default=3, metadata={"help": "Defines the average mask length."}
    )
    extra_padding_ratio: float = field(
        default=0.0,
        metadata={
            "help": (
                "Defines the ratio for the extra padding"
                "which are added only to the training data, in case of `span_infilling` uniformly."
            )
        },
    )
    conditional_generation: Optional[str] = field(
        default=None,
        metadata={
            "help": "It can be `span_infilling`, `prefix_lm`, `ul2`, or `ul2_with_unconditional`, `seq2seq`, `prefix_with_unconditional`"
            "In case of `span_infilling`: It trains/evals on filling spans like T5. In `prefix_lm`: it trains/evals"
            "on completing the prefixes like GPT2. In `ul2`, it trains on a mixture of span_infilling, agressive"
            "span_infilling, or prefix_lm and evals on prefix_lm with masking half of the sequence. In case of"
            "`ul2_with_unconditional`: it uses ul2 with also including unconditional generation during training."
            "`seq2seq` is used for translation or summarization tasks. `ul2_variable`: is ul2 for the different"
            "T5 mask_ratio till half of the sequence. `prefix_with_unconditional`: use prefix-lm with unconditional."
        },
    )
    eval_context_size: Optional[int] = field(
        default=None,
        metadata={
            "help": "By default we consider the half of sequence as prompt when evaluating for `conditional_generation` of"
            "`ul2` and `prefix_lm`. If this parameter is set, it specifies the context size during the evaluation."
        },
    )
    # TODO: later fix masking length with truncation.
    truncation_length: Optional[int] = field(
        default=0,
        metadata={
            "help": "If set, we will truncate the tokens from the end for the given length."
            "Note we still compute masking length based on original data length!"
        },
    )
    skip_special_tokens: bool = field(
        default=True,
        metadata={
            "help": "If training line by line set this to False to generate end token and cut. Also, in case you want to consider generation till </s> and cut the rest."
        },
    )
    # Parameters used in seq2seq training for summarization.
    """
    test_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input test data file to evaluate the metrics (rouge) on (a jsonlines or csv file)."},
    )
    """
    max_source_length: Optional[int] = field(
        default=1024,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    max_target_length: Optional[int] = field(
        default=128,
        metadata={
            "help": (
                "The maximum total sequence length for target text after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    val_max_target_length: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "The maximum total sequence length for validation target text after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded. Will default to `max_target_length`."
                "This argument is also used to override the ``max_length`` param of ``model.generate``, which is used "
                "during ``evaluate`` and ``predict``."
            )
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                "value if set."
            )
        },
    )
    num_beams: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
                "which is used during ``evaluate`` and ``predict``."
            )
        },
    )
    # Translation arguments.
    source_lang: str = field(
        default=None, metadata={"help": "Source language id for translation."}
    )
    target_lang: str = field(
        default=None, metadata={"help": "Target language id for translation."}
    )
    add_t5_tags: bool = field(
        default=False,
        metadata={
            "help": "In case of GLUE, it adds tags to the sentences like `sentence1:` ... ."
        },
    )
    # dataset verification
    verification_mode: str = field(
        default="basic_checks",
        metadata={
            "help": "Verification mode determining the checks to run on the downloaded/processed dataset information (checksums/size/splits/...)."
        },
    )
    streaming: bool = field(
        default=False,
        metadata={
            "help": "If set, we will stream the data from the disk or over the internet. This is useful for large datasets."
        },
    )
    shuffle: bool = field(
        default=True,
        metadata={"help": "If set, we will shuffle the data before training."},
    )
    min_train_seq_length: int = field(
        default=0,
        metadata={"help": "Minimum sequence length for train samples."},
    )
    min_eval_seq_length: int = field(
        default=0,
        metadata={"help": "Minimum sequence length for eval samples."},
    )

    def __post_init__(self):
        if (
            not self.tokenized_data_path
            and self.dataset_name is None
            and (self.train_file is None and self.validation_file is None)
        ):
            # NOTE: for pretraining, we detect whether we're on weka or nfs
            # and automatically set the dataset
            pass
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                if extension not in ["csv", "json", "txt"]:
                    raise ValueError(
                        "`train_file` should be a csv, a json or a txt file."
                    )
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                if extension not in ["csv", "json", "txt"]:
                    raise ValueError(
                        "`validation_file` should be a csv, a json or a txt file."
                    )

        if self.val_max_target_length is None:
            self.val_max_target_length = self.max_target_length

        if self.conditional_generation is not None:
            assert self.conditional_generation in [
                "span_infilling",
                "ul2",
                "ul2_with_unconditional",
                "prefix_lm",
                "prefix_with_unconditional",
                "seq2seq",
                "ul2_variable",
            ]

        tulu_flags = (
            self.is_tulu_pair,
            self.is_tulu_multiturn,
            self.is_tulu_sliding_window_multiturn,
        )
        # can only have at most 1 option toggled true
        assert sum(tulu_flags) < 2


@dataclass
class DiffusionArguments:
    """Defines the diffusion related parameters."""

    simplex_value: float = field(
        default=5.0,
        metadata={
            "help": (
                "We map the token ids to a vector of vocabulary size, where for tokens not"
                "equal to the token id `-simplex_value` is selected, and `simplex_value` otherwise."
            )
        },
    )
    num_diffusion_steps: int = field(
        default=2500, metadata={"help": "Defines the number of diffusion steps."}
    )
    num_inference_diffusion_steps: List[int] = field(
        default_factory=lambda: [1, 10, 100],
        metadata={"help": "Diffusion timesteps to try during inference."},
    )
    beta_schedule: str = field(
        default="squaredcos_improved_ddpm",
        metadata={
            "help": (
                "The beta schedule, a mapping from a beta range to a sequence of betas for stepping the model."
                "Choose from `linear`, `scaled_linear`, or `squaredcos_cap_v2`, `squaredcos_improved_ddpm`."
                "`squaredcos_improved_ddpm` model is proposed in eqn.17 in Improved ddpm"
                "(https://arxiv.org/pdf/2102.09672.pdf)"
            )
        },
    )
    sampling_type: str = field(
        default="top_p",
        metadata={"help": "Sampling type used during the logit projection."},
    )
    top_p: Optional[float] = field(
        default=None, metadata={"help": "top_p value for nucleus (top_p) sampling."}
    )
    clip_sample: bool = field(
        default=False,
        metadata={
            "help": "Whether to clip predicted sample between -1 and 1 for numerical stability in the noise scheduler."
        },
    )
    self_condition: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "If set, adds self-conditioning."
                "we consider the following options: `logits`: predicted logits, or `logits_with_projection`: to"
                "consider logits and apply the projection. After concatenating the inputs, we project inputs back"
                "with a projection layer to the half dimension. We also consider the cases of `logits_addition`"
                " and `logits_with_projection_addition` where we adds up the previous prediction to the logits,"
                "possibly with a projection operation. `logits_mean`: gets the average of logits and `logits_max`"
                "computes the maximum."
            )
        },
    )
    self_condition_mix_before_weights: bool = field(
        default=False,
        metadata={
            "help": "If set, mixes the softmax of simplexes and then apply the weights."
        },
    )
    self_condition_mix_logits_before_weights: bool = field(
        default=False,
        metadata={"help": "If set, mixes simplexes and then apply the weights."},
    )
    self_condition_mlp_projection: bool = field(
        default=False, metadata={"help": "If not set, uses a linear layer."}
    )
    self_condition_zeros_after_softmax: bool = field(
        default=False,
        metadata={
            "help": "If set, makes the softmax of previous_logits,"
            "in case previous_logits are zero, zero. This avoid extra bias introduced with using Linear[softmax(previous_logits), logits]"
        },
    )
    deepmind_conditional: bool = field(
        default=False,
        metadata={
            "help": "This is the way conditional is explained in the DeepMind paper"
            "https://arxiv.org/abs/2211.15089, figure 3. In this setup, we mask the self-conditioned, noisy, and original emebeddings,"
            "then we concat mask to these, and project all of them, and then add timestep embeddings."
        },
    )
    guidance_scale: float = field(
        default=1.0,
        metadata={
            "help": "classifier-free guidance is applied if guidance_scale > 1.0."
        },
    )
    classifier_free_uncond_input: str = field(
        default="empty_token",
        metadata={"help": "This can be one of `empty_token` or `noisy_simplex`."},
    )
    empty_token_be_mask: bool = field(
        default=False, metadata={"help": "If set, makes the empty token a mask."}
    )
    classifier_free_simplex_inputs: bool = field(
        default=False,
        metadata={
            "help": "If set to true, uses simplex representation for the unconditional input."
        },
    )
    temperature: float = field(
        default=1.0,
        metadata={"help": "Defines the softmax temperature before doing the sampling."},
    )
    guidance_softmax_combination: bool = field(
        default=True,
        metadata={"help": "If set, first applies softmax, then combines logits."},
    )
    generate_with_seed: bool = field(
        default=False, metadata={"help": "If set, generates with seed."}
    )
    multiply_factor: float = field(
        default=1.0,
        metadata={"help": "Determines the starting noise level."},
    )
    eval_dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the dataset to use for evaluation."},
    )
