"""Implements the metrics for evaluation of the diffusion models."""
from mauve import compute_mauve
from nltk.util import ngrams
import numpy as np
from collections import Counter
from scipy import stats
import operator

MAX_TEXT_LENGTH = 256


def mauve(predictions, references, featurize_model_name="gpt2-large"):
    """Computes MAUVE scores between two lists of generated text and reference text.
    Args:
    predictions (list of str) of predicttions.
    reference (list of str) of references.
    """
    results = compute_mauve(
        p_text=predictions,
        q_text=references,
        max_text_length=MAX_TEXT_LENGTH,
        featurize_model_name=featurize_model_name,
        verbose=False,
    )
    return results.mauve


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
    return np.nanmean(dist_1), np.nanmean(dist_2), np.nanmean(dist_3)


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
    return -a, -r, p
