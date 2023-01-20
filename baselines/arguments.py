from transformers import Seq2SeqTrainingArguments
from dataclasses import dataclass, field

@dataclass
class BaselineSeq2SeqTrainingArguments(Seq2SeqTrainingArguments):
    save_checkpoints_on_s3: bool = field(default=False, metadata={"help": "If set, instead of deleting the checkpoints when passing the limit of save checkpoints, it saves them on S3."})

