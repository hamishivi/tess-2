"""Implements the metrics for evaluation of the diffusion models."""
from mauve import compute_mauve
from nltk.util import ngrams
import numpy as np
from collections import Counter
from scipy import stats
import operator
import math
import scipy
import sklearn

MAX_TEXT_LENGTH = 256


def mauve(predictions, references, featurize_model_name="gpt2-large"):
    """Computes MAUVE scores between two lists of generated text and reference text.
    Args:
    predictions (list of str) of predictions.
    reference (list of str) of references.
    """
    results = compute_mauve(
        p_text=references,  # human-text.
        q_text=predictions,  # machine-text.
        max_text_length=MAX_TEXT_LENGTH,
        featurize_model_name=featurize_model_name,
        verbose=False,
        # These are the tricks to make `mauve` run faster if #examples > 5K.
        # See https://github.com/krishnap25/mauve#best-practices-for-mauve
        # num_buckets=500 if len(predictions) > 5000 else "auto",
        # kmeans_num_redo=1,
    )
    return {"muave": results.mauve}


def distinct_n_grams(texts):
    """Computes the average distinct n-grams of the generated texts.
    Args:
        texts (list of str): representing the generated texts.
    """
    dist_1, dist_2, dist_3 = [], [], []
    for text in texts:
        total_words = len(text.split())
        unigrams = set(ngrams(text.split(), 1))
        bigrams = set(ngrams(text.split(), 2))
        trigrams = set(ngrams(text.split(), 3))
        dist_1.append(len(unigrams) / total_words)
        dist_2.append(len(bigrams) / total_words)
        dist_3.append(len(trigrams) / total_words)
    return {"dist-1": np.nanmean(dist_1), "dist-2": np.nanmean(dist_2), "dist-3": np.nanmean(dist_3)}


def zipf(tokenized_texts, N=5000):
    """Computes the Zipf coefficient.

    Args:
        tokenized_texts (List[List[int]]) tokenized texts.
    Adapted from https://github.com/ari-holtzman/degen/blob/master/metrics/zipf.py
    """
    cnt = Counter()
    for tokenized_text in tokenized_texts:
        cnt.update(tokenized_text)

    xs = np.arange(1, min(len(cnt), N) + 1)
    ys = np.array(sorted(cnt.values(), key=operator.neg)[:N])
    a, b, r, p, std = stats.linregress(np.log(xs), np.log(ys))
    # Note that zipf_minus_a is the reported number.
    return {"zipf_minus_a": -a, "zipf_minus_r": -r, "zipf_p": p}


def accuracy(predictions, targets) -> dict:
    """Computes the average accuracy."""
    return {"accuracy": 100 * ((np.array(predictions) == np.array(targets)).mean())}


def pearson_corrcoef(predictions, targets) -> dict:
    """Computes Pearson correlation coefficient."""
    pearson_corrcoef = 100 * scipy.stats.pearsonr(targets, predictions)[0]

    # Note that if all the predictions will be the same, spearman
    # correlation is nan, to gaurad against this, we check the output
    # and return 0 in this case.
    if math.isnan(pearson_corrcoef):
        pearson_corrcoef = 0
    return {"pearson": pearson_corrcoef}


def spearman_corrcoef(predictions, targets) -> dict:
    """Computes Spearman correlation coefficient."""
    spearman_corrcoef = 100 * scipy.stats.spearmanr(targets, predictions)[0]

    # Note that if all the predictions will be the same, spearman
    # correlation is nan, to gaurad against this, we check the output
    # and return 0 in this case.
    if math.isnan(spearman_corrcoef):
        spearman_corrcoef = 0
    return {"spearmanr": spearman_corrcoef}


def f1_score_with_invalid(predictions, targets) -> dict:
    """Computes F1 score,  with any prediction != 0 or 1 is counted as incorrect.
    Args:
      targets: list of targets, either 0 or 1
      predictions: list of predictions, any integer value
    Returns:
      F1 score, where any prediction != 0 or 1 is counted as wrong.
    """
    targets, predictions = np.asarray(targets), np.asarray(predictions)
    # Get indices of invalid predictions.
    invalid_idx_mask = np.logical_and(predictions != 0, predictions != 1)
    # For any prediction != 0 or 1, we set the prediction to the opposite of its corresponding target.
    predictions[invalid_idx_mask] = 1 - targets[invalid_idx_mask]
    return {"f1": 100 * sklearn.metrics.f1_score(targets, predictions)}


# TODO: maybe gaurd against invalid values https://stackoverflow.com/questions/56865344/how-do-i-calculate-the-matthews-correlation-coefficient-in-tensorflow
def matthews_corrcoef(predictions, targets) -> dict:
    """Computes the Matthews correlation coefficient."""
    return {"matthews_correlation": 100 * sklearn.metrics.matthews_corrcoef(targets, predictions)}


def get_glue_metrics(task):
    GLUE_TASKS_TO_METRICS = {
        "mrpc": [f1_score_with_invalid, accuracy],
        "cola": [matthews_corrcoef],
        "sst2": [accuracy],
        "stsb": [pearson_corrcoef, spearman_corrcoef],
        "qqp": [f1_score_with_invalid, accuracy],
        "mnli": [accuracy],
        "qnli": [accuracy],
        "rte": [accuracy],
        "wnli": [accuracy],
    }
    return GLUE_TASKS_TO_METRICS[task]
