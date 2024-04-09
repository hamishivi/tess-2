"""
Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...) on a text file or a dataset.
"""

import logging
import os
import sys

import datasets
import evaluate
import torch
import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    set_seed,
)
from transformers.trainer_callback import TrainerState
from transformers.trainer_utils import get_last_checkpoint

from .arguments import get_args
from .data.data_collator import DataCollatorForSeq2Seq
from .data.data_utils import load_data
from .data.postprocessors import postprocess_text_for_metric
from .inference.inference_utils import process_text
from .models.llama.modeling_llama import LlamaForSeq2SeqLM
from .trainer_ar import ARTrainer

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
# check_min_version("4.40.0.dev0")
# require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")
logger = logging.getLogger(__name__)


MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


# @dataclass
# class ModelArguments:
#     """
#     Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
#     """

#     model_name_or_path: Optional[str] = field(
#         default=None,
#         metadata={
#             "help": (
#                 "The model checkpoint for weights initialization. Don't set if you want to train a model from scratch."
#             )
#         },
#     )
#     model_type: Optional[str] = field(
#         default=None,
#         metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
#     )
#     config_overrides: Optional[str] = field(
#         default=None,
#         metadata={
#             "help": (
#                 "Override some existing default config settings when a model is trained from scratch. Example: "
#                 "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
#             )
#         },
#     )
#     config_name: Optional[str] = field(
#         default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
#     )
#     tokenizer_name: Optional[str] = field(
#         default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
#     )
#     cache_dir: Optional[str] = field(
#         default=None,
#         metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
#     )
#     use_fast_tokenizer: bool = field(
#         default=True,
#         metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
#     )
#     model_revision: str = field(
#         default="main",
#         metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
#     )
#     token: str = field(
#         default=None,
#         metadata={
#             "help": (
#                 "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
#                 "generated when running `huggingface-cli login` (stored in `~/.huggingface`)."
#             )
#         },
#     )
#     use_auth_token: bool = field(
#         default=None,
#         metadata={
#             "help": "The `use_auth_token` argument is deprecated and will be removed in v4.34. Please use `token` instead."
#         },
#     )
#     trust_remote_code: bool = field(
#         default=False,
#         metadata={
#             "help": (
#                 "Whether or not to allow for custom models defined on the Hub in their own modeling files. This option "
#                 "should only be set to `True` for repositories you trust and in which you have read the code, as it will "
#                 "execute code present on the Hub on your local machine."
#             )
#         },
#     )
#     torch_dtype: Optional[str] = field(
#         default=None,
#         metadata={
#             "help": (
#                 "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
#                 "dtype will be automatically derived from the model's weights."
#             ),
#             "choices": ["auto", "bfloat16", "float16", "float32"],
#         },
#     )
#     low_cpu_mem_usage: bool = field(
#         default=False,
#         metadata={
#             "help": (
#                 "It is an option to create the model as an empty shell, then only materialize its parameters when the pretrained weights are loaded. "
#                 "set True will benefit LLM loading time and RAM consumption."
#             )
#         },
#     )

#     def __post_init__(self):
#         if self.config_overrides is not None and (self.config_name is not None or self.model_name_or_path is not None):
#             raise ValueError(
#                 "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
#             )


# @dataclass
# class DataTrainingArguments:
#     """
#     Arguments pertaining to what data we are going to input our model for training and eval.
#     """

#     dataset_name: Optional[str] = field(
#         default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
#     )
#     dataset_config_name: Optional[str] = field(
#         default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
#     )
#     train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
#     validation_file: Optional[str] = field(
#         default=None,
#         metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
#     )
#     max_train_samples: Optional[int] = field(
#         default=None,
#         metadata={
#             "help": (
#                 "For debugging purposes or quicker training, truncate the number of training examples to this "
#                 "value if set."
#             )
#         },
#     )
#     max_eval_samples: Optional[int] = field(
#         default=None,
#         metadata={
#             "help": (
#                 "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
#                 "value if set."
#             )
#         },
#     )
#     streaming: bool = field(default=False, metadata={"help": "Enable streaming mode"})
#     block_size: Optional[int] = field(
#         default=None,
#         metadata={
#             "help": (
#                 "Optional input sequence length after tokenization. "
#                 "The training dataset will be truncated in block of this size for training. "
#                 "Default to the model max input length for single sentence inputs (take into account special tokens)."
#             )
#         },
#     )
#     overwrite_cache: bool = field(
#         default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
#     )
#     validation_split_percentage: Optional[int] = field(
#         default=5,
#         metadata={
#             "help": "The percentage of the train set used as validation set in case there's no validation split"
#         },
#     )
#     preprocessing_num_workers: Optional[int] = field(
#         default=None,
#         metadata={"help": "The number of processes to use for the preprocessing."},
#     )
#     keep_linebreaks: bool = field(
#         default=True, metadata={"help": "Whether to keep line breaks when using TXT files or not."}
#     )

#     def __post_init__(self):
#         if self.streaming:
#             require_version("datasets>=2.0.0", "The streaming feature requires `datasets>=2.0.0`")

#         if self.dataset_name is None and self.train_file is None and self.validation_file is None:
#             raise ValueError("Need either a dataset name or a training/validation file.")
#         else:
#             if self.train_file is not None:
#                 extension = self.train_file.split(".")[-1]
#                 assert extension in ["csv", "json", "txt"], "`train_file` should be a csv, a json or a txt file."
#             if self.validation_file is not None:
#                 extension = self.validation_file.split(".")[-1]
#                 assert extension in ["csv", "json", "txt"], "`validation_file` should be a csv, a json or a txt file."


summarization_name_mapping = {
    "amazon_reviews_multi": ("review_body", "review_title"),
    "big_patent": ("description", "abstract"),
    "cnn_dailymail": ("article", "highlights"),
    "orange_sum": ("text", "summary"),
    "pn_summary": ("article", "summary"),
    "psc": ("extract_text", "summary_text"),
    "samsum": ("dialogue", "summary"),
    "thaisum": ("body", "summary"),
    "xglue": ("news_body", "news_title"),
    "xsum": ("document", "summary"),
    "wiki_summary": ("article", "highlights"),
    "multi_news": ("document", "summary"),
}


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    # parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    # if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
    #     # If we pass only one argument to the script and it's the path to a json file,
    #     # let's parse it to get our arguments.
    #     model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    # else:
    #     model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # if model_args.use_auth_token is not None:
    #     warnings.warn(
    #         "The `use_auth_token` argument is deprecated and will be removed in v4.34. Please use `token` instead.",
    #         FutureWarning,
    #     )
    #     if model_args.token is not None:
    #         raise ValueError("`token` and `use_auth_token` are both specified. Please set only the argument `token`.")
    #     model_args.token = model_args.use_auth_token

    # # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # # information sent is the one passed as arguments along with your Python/PyTorch versions.
    # send_example_telemetry("run_clm", model_args, data_args)
    # parse args
    model_args, data_args, training_args, diffusion_args = get_args()

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

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    # if data_args.dataset_name is not None:
    #     # Downloading and loading a dataset from the hub.
    #     raw_datasets = load_dataset(
    #         data_args.dataset_name,
    #         data_args.dataset_config_name,
    #         cache_dir=model_args.cache_dir,
    #         token=model_args.token,
    #         streaming=data_args.streaming,
    #     )
    #     if "validation" not in raw_datasets.keys():
    #         raw_datasets["validation"] = load_dataset(
    #             data_args.dataset_name,
    #             data_args.dataset_config_name,
    #             split=f"train[:{data_args.validation_split_percentage}%]",
    #             cache_dir=model_args.cache_dir,
    #             token=model_args.token,
    #             streaming=data_args.streaming,
    #         )
    #         raw_datasets["train"] = load_dataset(
    #             data_args.dataset_name,
    #             data_args.dataset_config_name,
    #             split=f"train[{data_args.validation_split_percentage}%:]",
    #             cache_dir=model_args.cache_dir,
    #             token=model_args.token,
    #             streaming=data_args.streaming,
    #         )
    # else:
    #     data_files = {}
    #     dataset_args = {}
    #     if data_args.train_file is not None:
    #         data_files["train"] = data_args.train_file
    #     if data_args.validation_file is not None:
    #         data_files["validation"] = data_args.validation_file
    #     extension = (
    #         data_args.train_file.split(".")[-1]
    #         if data_args.train_file is not None
    #         else data_args.validation_file.split(".")[-1]
    #     )
    #     if extension == "txt":
    #         extension = "text"
    #         dataset_args["keep_linebreaks"] = data_args.keep_linebreaks
    #     raw_datasets = load_dataset(
    #         extension,
    #         data_files=data_files,
    #         cache_dir=model_args.cache_dir,
    #         token=model_args.token,
    #         **dataset_args,
    #     )
    #     # If no validation data is there, validation_split_percentage will be used to divide the dataset.
    #     if "validation" not in raw_datasets.keys():
    #         raw_datasets["validation"] = load_dataset(
    #             extension,
    #             data_files=data_files,
    #             split=f"train[:{data_args.validation_split_percentage}%]",
    #             cache_dir=model_args.cache_dir,
    #             token=model_args.token,
    #             **dataset_args,
    #         )
    #         raw_datasets["train"] = load_dataset(
    #             extension,
    #             data_files=data_files,
    #             split=f"train[{data_args.validation_split_percentage}%:]",
    #             cache_dir=model_args.cache_dir,
    #             token=model_args.token,
    #             **dataset_args,
    #         )

    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    # load data
    raw_datasets = load_data(data_args, model_args)

    # TODO: add flash attention
    config_kwargs = {
        # "cache_dir": model_args.cache_dir,
        # "revision": model_args.model_revision,
        # "token": model_args.token,
        # "trust_remote_code": model_args.trust_remote_code,
    }
    # if model_args.config_name:
    #     config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    if model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(
            model_args.model_name_or_path, **config_kwargs
        )
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")
        if model_args.config_overrides is not None:
            logger.info(f"Overriding config: {model_args.config_overrides}")
            config.update_from_string(model_args.config_overrides)
            logger.info(f"New config: {config}")

    tokenizer_kwargs = {
        # "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        # "revision": model_args.model_revision,
        # "token": model_args.token,
        # "trust_remote_code": model_args.trust_remote_code,
    }
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name, **tokenizer_kwargs
        )
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path, **tokenizer_kwargs
        )
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script. "
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    assert tokenizer.padding_side == "right"
    try:
        tokenizer.add_eos_token = True
    except AttributeError:
        # roberta does not have this
        pass
    if not tokenizer.pad_token_id:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    if model_args.model_name_or_path:
        # torch_dtype = (
        #     model_args.torch_dtype
        #     if model_args.torch_dtype in ["auto", None]
        #     else getattr(torch, model_args.torch_dtype)
        # )
        # identify dtype
        torch_dtype = torch.float32
        if training_args.bf16:
            torch_dtype = torch.bfloat16
        elif training_args.fp16:
            torch_dtype = torch.float16
        model = LlamaForSeq2SeqLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            # cache_dir=model_args.cache_dir,
            # revision=model_args.model_revision,
            # token=model_args.token,
            # trust_remote_code=model_args.trust_remote_code,
            torch_dtype=torch_dtype,
            # low_cpu_mem_usage=model_args.low_cpu_mem_usage,
        )
    else:
        model = AutoModelForCausalLM.from_config(
            config, trust_remote_code=model_args.trust_remote_code
        )
        n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
        logger.info(
            f"Training new model from scratch - Total size={n_params/2**20:.2f}M params"
        )

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    vocab_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > vocab_size:
        model.resize_token_embeddings(len(tokenizer))
        model.config.pad_token_id = tokenizer.pad_token_id

    total_seq2seq_length = data_args.max_source_length + data_args.max_target_length
    if (
        hasattr(model.config, "max_position_embeddings")
        and model.config.max_position_embeddings < total_seq2seq_length
    ):
        if model_args.resize_position_embeddings is None:
            logger.warning(
                "Increasing the model's number of position embedding vectors from"
                f" {model.config.max_position_embeddings} to {total_seq2seq_length}."
            )
            # position_ids starts from `padding_idx + 1` (padding_index=1) and we therefore requires
            # 2 more position embeddings.
            model.resize_position_embeddings(
                total_seq2seq_length + 2,
                with_alternatation=model_args.resize_position_embeddings_alternatively,
            )
        elif model_args.resize_position_embeddings:
            model.resize_position_embeddings(
                total_seq2seq_length + 2,
                with_alternatation=model_args.resize_position_embeddings_alternatively,
            )
        else:
            raise ValueError(
                f"`max_source_length`+`max_target_length` is set to {total_seq2seq_length}, but the model only has"
                f" {model.config.max_position_embeddings} position encodings. Consider either reducing"
                f" `max_source_length`+`max_target_length` to {model.config.max_position_embeddings} or to automatically resize the"
                " model's position encodings by passing `--resize_position_embeddings`."
            )

    # Preprocessing the datasets.
    # # First we tokenize all the texts.
    # if training_args.do_train:
    #     column_names = list(raw_datasets["train"].features)
    # else:
    #     column_names = list(raw_datasets["validation"].features)
    # text_column_name = "text" if "text" in column_names else column_names[0]
    if training_args.do_train:
        column_names = raw_datasets["train"].column_names
    elif training_args.do_eval:
        column_names = raw_datasets["validation"].column_names
    elif training_args.do_predict:
        column_names = raw_datasets["test"].column_names
    else:
        logger.info(
            "There is nothing to do. Please pass `do_train`, `do_eval` and/or `do_predict`."
        )
        return

    # # since this will be pickled to avoid _LazyModule error in Hasher force logger loading before tokenize_function
    # tok_logger = transformers.utils.logging.get_logger("transformers.tokenization_utils_base")

    # Get the column names for input/target.
    dataset_columns = summarization_name_mapping.get(data_args.dataset_name, None)
    assert dataset_columns is not None, "You need to provide the columns names."
    text_column, summary_column = dataset_columns[0], dataset_columns[1]

    # Temporarily set max_target_length for training.
    max_target_length = data_args.max_target_length

    # def tokenize_function(examples):
    #     with CaptureLogger(tok_logger) as cl:
    #         output = tokenizer(examples[text_column_name])
    #     # clm input could be much much longer than block_size
    #     if "Token indices sequence length is longer than the" in cl.out:
    #         tok_logger.warning(
    #             "^^^^^^^^^^^^^^^^ Please ignore the warning above - this long input will be chunked into smaller bits"
    #             " before being passed to the model."
    #         )
    #     return output

    def preprocess_function(examples):
        # remove pairs where at least one record is None

        inputs, targets = [], []
        for i in range(len(examples[text_column])):
            if examples[text_column][i] and examples[summary_column][i]:
                inputs.append(examples[text_column][i])
                targets.append(examples[summary_column][i])
        # TODO: we need to process first the target, then cut the inputs to the max_length-target length to use the
        # maximum number of tokens.
        model_inputs = tokenizer(
            inputs,
            max_length=data_args.max_source_length,
            padding=False,
            truncation=True,
        )
        # Tokenize targets with the `text_target` keyword argument
        labels = tokenizer(
            text_target=targets,
            max_length=max_target_length,
            padding=False,
            truncation=True,
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    # with training_args.main_process_first(desc="dataset map tokenization"):
    #     if not data_args.streaming:
    #         tokenized_datasets = raw_datasets.map(
    #             tokenize_function,
    #             batched=True,
    #             num_proc=data_args.preprocessing_num_workers,
    #             remove_columns=column_names,
    #             load_from_cache_file=not data_args.overwrite_cache,
    #             desc="Running tokenizer on dataset",
    #         )
    #     else:
    #         tokenized_datasets = raw_datasets.map(
    #             tokenize_function,
    #             batched=True,
    #             remove_columns=column_names,
    #         )
    # if hasattr(config, "max_position_embeddings"):
    #     max_pos_embeddings = config.max_position_embeddings
    # else:
    #     # Define a default value if the attribute is missing in the config.
    #     max_pos_embeddings = 1024

    # if data_args.block_size is None:
    #     block_size = tokenizer.model_max_length
    #     if block_size > max_pos_embeddings:
    #         logger.warning(
    #             f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
    #             f"Using block_size={min(1024, max_pos_embeddings)} instead. You can change that default value by passing --block_size xxx."
    #         )
    #         if max_pos_embeddings > 0:
    #             block_size = min(1024, max_pos_embeddings)
    #         else:
    #             block_size = 1024
    # else:
    #     if data_args.block_size > tokenizer.model_max_length:
    #         logger.warning(
    #             f"The block_size passed ({data_args.block_size}) is larger than the maximum length for the model "
    #             f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
    #         )
    #     block_size = min(data_args.block_size, tokenizer.model_max_length)

    # # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
    # def group_texts(examples):
    #     # Concatenate all texts.
    #     concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    #     total_length = len(concatenated_examples[list(examples.keys())[0]])
    #     # We drop the small remainder, and if the total_length < block_size  we exclude this batch and return an empty dict.
    #     # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
    #     total_length = (total_length // block_size) * block_size
    #     # Split by chunks of max_len.
    #     result = {
    #         k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
    #         for k, t in concatenated_examples.items()
    #     }
    #     result["labels"] = result["input_ids"].copy()
    #     return result

    # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a remainder
    # for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value might be slower
    # to preprocess.
    #
    # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
    # https://huggingface.co/docs/datasets/process#map

    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
        with training_args.main_process_first(desc="train dataset map pre-processing"):
            train_dataset = train_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on train dataset",
            )

    if training_args.do_eval:
        max_target_length = data_args.val_max_target_length
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))
        with training_args.main_process_first(
            desc="validation dataset map pre-processing"
        ):
            eval_dataset = eval_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on validation dataset",
            )

    def preprocess_logits_for_metrics(logits):
        return logits.argmax(dim=-1)

    if training_args.do_predict:
        max_target_length = data_args.val_max_target_length
        if "test" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        test_dataset = raw_datasets["test"]
        if data_args.max_predict_samples is not None:
            max_predict_samples = min(len(test_dataset), data_args.max_predict_samples)
            test_dataset = test_dataset.select(range(max_predict_samples))
        with training_args.main_process_first(
            desc="prediction dataset map pre-processing"
        ):
            test_dataset = test_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on prediction dataset",
            )

    # Data collator. To be consistent with the run_mlm.py we need to add `mode`.
    data_collator = lambda mode: DataCollatorForSeq2Seq(  # noqa: E731
        tokenizer,
        # Note that if you do not use `pad_to_max_length`, this becomes very slow on multi-gpus.
        padding="max_length" if data_args.pad_to_max_length else True,
        max_length=data_args.max_seq_length,
        pad_to_multiple_of=8 if training_args.fp16 else None,
    )

    # with training_args.main_process_first(desc="grouping texts together"):
    #     if not data_args.streaming:
    #         lm_datasets = tokenized_datasets.map(
    #             group_texts,
    #             batched=True,
    #             num_proc=data_args.preprocessing_num_workers,
    #             load_from_cache_file=not data_args.overwrite_cache,
    #             desc=f"Grouping texts in chunks of {block_size}",
    #         )
    #     else:
    #         lm_datasets = tokenized_datasets.map(
    #             group_texts,
    #             batched=True,
    #         )

    # if training_args.do_train:
    #     if "train" not in tokenized_datasets:
    #         raise ValueError("--do_train requires a train dataset")
    #     train_dataset = lm_datasets["train"]
    #     if data_args.max_train_samples is not None:
    #         max_train_samples = min(len(train_dataset), data_args.max_train_samples)
    #         train_dataset = train_dataset.select(range(max_train_samples))

    # if training_args.do_eval:
    #     if "validation" not in tokenized_datasets:
    #         raise ValueError("--do_eval requires a validation dataset")
    #     eval_dataset = lm_datasets["validation"]
    #     if data_args.max_eval_samples is not None:
    #         max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
    #         eval_dataset = eval_dataset.select(range(max_eval_samples))

    #     def preprocess_logits_for_metrics(logits, labels):
    #         if isinstance(logits, tuple):
    #             # Depending on the model and config, logits may contain extra tensors,
    #             # like past_key_values, but logits always come first
    #             logits = logits[0]
    #         return logits.argmax(dim=-1)

    #     metric = evaluate.load("accuracy", cache_dir=model_args.cache_dir)

    #     def compute_metrics(eval_preds):
    #         preds, labels = eval_preds
    #         # preds have the same shape as the labels, after the argmax(-1) has been calculated
    #         # by preprocess_logits_for_metrics but we need to shift the labels
    #         labels = labels[:, 1:].reshape(-1)
    #         preds = preds[:, :-1].reshape(-1)
    #         return metric.compute(predictions=preds, references=labels)

    # Metric
    metric = evaluate.load("rouge")

    def compute_metrics(results):
        keys = ["pred_texts_from_simplex_masked", "pred_texts_from_logits_masked"]
        metrics = {}
        for key in keys:
            decoded_preds = (
                process_text(results[key])
                if not data_args.skip_special_tokens
                else results[key]
            )
            # Note that since decoded_labels is getting updated after post-process, we
            # need to compute it here for each key.
            decoded_labels = (
                process_text(results["gold_texts_masked"])
                if not data_args.skip_special_tokens
                else results["gold_texts_masked"]
            )
            decoded_preds, decoded_labels = postprocess_text_for_metric(
                "rouge", decoded_preds, decoded_labels
            )
            key_metrics = metric.compute(
                predictions=decoded_preds, references=decoded_labels, use_stemmer=True
            )
            key_metrics = {k: round(v * 100, 4) for k, v in key_metrics.items()}
            key_metrics = {f"{key}_{k}": v for k, v in key_metrics.items()}
            metrics.update(key_metrics)
        return metrics

    # Initialize our Trainer
    trainer = ARTrainer(
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
        data_args=data_args,
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
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()

        max_eval_samples = (
            data_args.max_eval_samples
            if data_args.max_eval_samples is not None
            else len(eval_dataset)
        )
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
        # try:
        #     perplexity = math.exp(metrics["eval_loss"])
        # except OverflowError:
        #     perplexity = float("inf")
        # metrics["perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.do_predict:
        logger.info("*** Test ***")
        metrics = trainer.evaluate(test_dataset, metric_key_prefix="test")
        max_predict_samples = (
            data_args.max_predict_samples
            if data_args.max_predict_samples is not None
            else len(test_dataset)
        )
        metrics["test_samples"] = min(max_predict_samples, len(test_dataset))
        trainer.log_metrics("test", metrics)
        trainer.save_metrics("test", metrics)
    # kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "text-generation"}
    # if data_args.dataset_name is not None:
    #     kwargs["dataset_tags"] = data_args.dataset_name
    #     if data_args.dataset_config_name is not None:
    #         kwargs["dataset_args"] = data_args.dataset_config_name
    #         kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
    #     else:
    #         kwargs["dataset"] = data_args.dataset_name

    # if training_args.push_to_hub:
    #     trainer.push_to_hub(**kwargs)
    # else:
    #     trainer.create_model_card(**kwargs)


# def _mp_fn(index):
#     # For xla_spawn (TPUs)
#     main()


if __name__ == "__main__":
    main()
