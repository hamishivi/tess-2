import pdb
import json
import numpy as np

path = "/net/nfs.cirrascale/s2-research/rabeehk/outputs/paper_experiments/ours_eval/self-condition-original_0.99_56_1000/eval_results.json"

name_to_short = {
    "pred_texts_from_simplex_masked_perplexity": "PPL",
    "pred_texts_from_simplex_masked_dist-1": "Dist-1",
    "pred_texts_from_simplex_masked_dist-2": "Dist-2",
    "pred_texts_from_simplex_masked_dist-3": "Dist-3",
    "pred_texts_from_simplex_masked_muave": "MAUVE",
    "pred_texts_from_simplex_masked_repetition": "Repetition",
    "pred_texts_from_simplex_masked_zipf_minus_a": "ZIPF-a",
    "pred_texts_from_simplex_masked_zipf_minus_r": "ZIPF-r",
    "pred_texts_from_simplex_masked_zipf_p": "ZIPF-p",
    "pred_texts_from_logits_masked_perplexity": "PPL",
    "pred_texts_from_logits_masked_dist-1": "Dist-1",
    "pred_texts_from_logits_masked_dist-2": "Dist-2",
    "pred_texts_from_logits_masked_dist-3": "Dist-3",
    "pred_texts_from_logits_masked_muave": "MAUVE",
    "pred_texts_from_logits_masked_repetition": "Repetition",
    "pred_texts_from_logits_masked_zipf_minus_a": "ZIPF-a",
    "pred_texts_from_logits_masked_zipf_minus_r": "ZIPF-r",
    "pred_texts_from_logits_masked_zipf_p": "ZIPF-p",
}

ordered_key = ["MAUVE", "PPL", "Dist-1", "Dist-2", "Dist-3", "ZIPF-a", "Repetition"]
metrics = json.load(open(f"{path}"))
metrics = {
    name_to_short[k]: np.round(100 * v, 2) if not name_to_short[k] in ["PPL", "ZIPF-a", "Repetition"] else np.round(v, 2)
    for k, v in metrics.items()
}

values = []
for k in ordered_key:
    values.append(metrics[k])
values = [str(v) for v in values]
values = "&".join(values)
print(ordered_key)
print(values)
print("_" * 100)
