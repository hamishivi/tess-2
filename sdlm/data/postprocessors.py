import nltk  # Here to have a nice missing dependency error message early on
from transformers.utils import is_offline_mode
from filelock import FileLock

try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    if is_offline_mode():
        raise LookupError("Offline mode: run this script without TRANSFORMERS_OFFLINE first to download nltk data files")
    with FileLock(".lock") as lock:
        nltk.download("punkt", quiet=True)

def string_to_float(string, default=-1.):
    """Converts string to float, using default when conversion not possible."""
    try:
        return float(string)
    except ValueError:
        return default


def string_to_int(string, default=-1):
    """Converts string to int, using default when conversion not possible."""
    try:
        return int(string)
    except ValueError:
        return default


def get_post_processor(task):
    """Returns post processor required to apply on the predictions/targets
    before computing metrics for each task."""
    if task == "stsb":
        return string_to_float
    elif task in ["qqp", "cola", "mrpc"]:
        return string_to_int
    else:
        return None


def postprocess_text_for_metric(metric, preds, labels=None, sources=None):
    if metric == "sari":
        assert sources is not None
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]
        sources = [source.strip() for source in sources]
        return preds, labels, sources
    elif metric == "rouge":
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]
        # rougeLSum expects newline after each sentence
        preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
        labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]
        return preds, labels
    elif metric == "bleu":
        preds = [pred.strip() for pred in preds]
        labels = [[label.strip()] for label in labels]
        return preds, labels
    elif metric in ["bertscore", "bertscore_them"]:
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]
        return preds, labels
    elif metric in ["dist"]:
        preds = [pred.strip() for pred in preds]
        return preds
    else:
        raise NotImplementedError