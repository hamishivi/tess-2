"""Arguments used in training/inference/data processing."""
from dataclasses import dataclass, field
from typing import Optional

from transformers import MODEL_MAPPING, SchedulerType
from transformers import TrainingArguments as HFTrainingArguments

MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": ("The model checkpoint for weights initialization. Don't set if you want to train a model from scratch.")
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
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
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
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

    def __post_init__(self):
        if self.config_overrides is not None and (self.config_name is not None or self.model_name_or_path is not None):
            raise ValueError("--config_overrides can't be used in combination with --config_name or --model_name_or_path")


@dataclass
class TrainingArguments(HFTrainingArguments):
    per_device_train_batch_size: int = field(
        default=8, metadata={"help": "Batch size (per device) for the training dataloader."}
    )
    per_device_eval_batch_size: int = field(
        default=8, metadata={"help": "Batch size (per device) for the evaluation dataloader."}
    )
    learning_rate: float = field(
        default=5e-5, metadata={"help": "Initial learning rate (after the potential warmup period) to use."}
    )
    weight_decay: float = field(default=0.0, metadata={"help": "Weight decay to use."})
    num_train_epochs: int = field(default=3, metadata={"help": "Total number of training epochs to perform."})
    max_train_steps: Optional[int] = field(
        default=None,
        metadata={"help": "Total number of training steps to perform. If provided, overrides num_train_epochs."},
    )
    gradient_accumulation_steps: int = field(
        default=1, metadata={"help": "Number of updates steps to accumulate before performing a backward/update pass."}
    )
    lr_scheduler_type: SchedulerType = field(
        default="linear",
        metadata={
            "help": (
                "The scheduler type to use. It can be `linear`, `cosine`,"
                "`cosine_with_restarts`, `polynomial`, `constant`, and `constant_with_warmup`"
            )
        },
    )
    num_warmup_steps: int = field(default=0, metadata={"help": "Number of steps for the warmup in the lr scheduler."})
    output_dir: Optional[str] = field(default=None, metadata={"help": "Where to store the final model."})
    seed: Optional[int] = field(default=42, metadata={"help": "A seed for reproducible training."})
    checkpointing_steps: int = field(default=1000, metadata={"help": "Specifies the checkpoint step."})
    resume_from_checkpoint: Optional[str] = field(
        default=None, metadata={"help": "If the training should continue from a checkpoint folder."}
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    tokenized_data_path: Optional[str] = field(default=None, metadata={"help": "If set, reads a tokenized train data."})
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    overwrite_cache: bool = field(default=False, metadata={"help": "Overwrite the cached training and evaluation sets"})
    validation_split_ratio: Optional[float] = field(
        default=0.01,
        metadata={"help": "The ratio(< 1.0) of the train set used as validation set in case there's no validation split."},
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
    mlm_probability: float = field(
        default=0.15, metadata={"help": "Ratio of tokens to mask for masked language modeling loss"}
    )
    line_by_line: bool = field(
        default=False,
        metadata={"help": "Whether distinct lines of text in the dataset are to be handled as distinct sequences."},
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

    def __post_init__(self):
        if (
            not self.tokenized_data_path
            and self.dataset_name is None
            and self.train_file is None
            and self.validation_file is None
        ):
            raise ValueError("Need either a dataset name or a training/validation file or a tokenized dataset path.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                if extension not in ["csv", "json", "txt"]:
                    raise ValueError("`train_file` should be a csv, a json or a txt file.")
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                if extension not in ["csv", "json", "txt"]:
                    raise ValueError("`validation_file` should be a csv, a json or a txt file.")


@dataclass
class DiffusionArguments:
    """Defines the diffusion related parameters."""

    simplex_value: int = field(
        default=5,
        metadata={
            "help": (
                "We map the token ids to a vector of vocabulary size, where for tokens not"
                "equal to the token id `-simplex_value` is selected, and `simplex_value` otherwise."
            )
        },
    )
    num_diffusion_steps: int = field(default=2500, metadata={"help": "Defines the number of diffusion steps."})
    beta_schedule: str = field(
        default="squaredcos_cap_v2",
        metadata={
            "help": (
                "The beta schedule, a mapping from a beta range to a sequence of betas for stepping the model."
                "Choose from `linear`, `scaled_linear`, or `squaredcos_cap_v2`."
            )
        },
    )
    predict_epsilon: bool = field(
        default=False,
        metadata={"help": "Uses for scheduler, if model predicts the noise (epsilon), or the samples instead of the noise."},
    )
    sampling_type: str = field(default="top_p", metadata={"help": "Sampling type used during the logit projection."})
    top_p: float = field(default=0.95, metadata={"help": "top_p value for nucleus (top_p) sampling."})
