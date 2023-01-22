from transformers import Seq2SeqTrainingArguments
from dataclasses import dataclass, field

@dataclass
class BaselineSeq2SeqTrainingArguments(Seq2SeqTrainingArguments):
    save_checkpoints_on_s3: bool = field(default=False, metadata={"help": "If set, instead of deleting the checkpoints when passing the limit of save checkpoints, it saves them on S3."})
    load_states_in_eval_from_model_path: bool = field(
        default=True,
        metadata={
            "help": "In case of only using --do_eval without --do_train, use it to load the states before eval."
            "keep this to true, it causes otherwise an issue with huggingface when doing only --do_eval."
        },
    )
