from transformers import Seq2SeqTrainingArguments
from dataclasses import dataclass, field
from typing import Optional

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
    top_p: Optional[float] = field(default=None, metadata={"help": "top_p value for nucleus (top_p) sampling."})
    temperature: float = field(default=1.0, metadata={"help": "Defines the softmax temperature before doing the sampling."})

