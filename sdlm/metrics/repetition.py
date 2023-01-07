"""Computes the repetition metric. Adapted from: https://raw.githubusercontent.com/ari-holtzman/degen/master/metrics/repetition.py"""
import pdb


def repetition(tokenized_texts, tokenizer):
    """
    Args:
        tokenized_texts: (List[List[int]]) generated input tokenized texts.

    Computes the repetition metric https://arxiv.org/pdf/1904.09751.pdf showing how each
    example is repeating itself, specifically the phrase the generation is repeating
    and how many times it is repeated.
    """
    SEP = tokenizer.encode(tokenizer.bos_token)[0]
    repetition_stats = []
    max_n = 90
    num_examples = len(tokenized_texts)
    n_repeated_examples = 0
    for tokenized_text in tokenized_texts:
        if tokenized_text[-1] == SEP:
            tokenized_text.pop(-1)
        rev_gen = list(reversed(tokenized_text))
        last_n_repeats = [0] * max_n
        for n in range(1, max_n + 1):
            n_repeat = 1
            while (
                len(rev_gen[n * n_repeat : n * (n_repeat + 1)]) == n
                and rev_gen[n * n_repeat : n * (n_repeat + 1)] == rev_gen[:n]
            ):
                n_repeat += 1
            last_n_repeats[n - 1] = n_repeat
        max_repeated_n = max(range(max_n), key=lambda x: last_n_repeats[x])
        if last_n_repeats[max_repeated_n] > 1 and (max_repeated_n + 1 >= 3 or last_n_repeats[max_repeated_n] > 50):
            repetition_stats.append(
                {
                    "repeated_phrase": list(reversed(rev_gen[: max_repeated_n + 1])),
                    "repeated_times": last_n_repeats[max_repeated_n],
                    "repeated_phrase_length": max_repeated_n + 1,
                }
            )
            n_repeated_examples += 1
        else:
            repetition_stats.append({})

    return {"repetition": n_repeated_examples * 1.0 / num_examples}  # , "repetition_stats": repetition_stats}
