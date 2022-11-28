"""Implements the metrics for evaluation of the diffusion models."""
from mauve import compute_mauve

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
