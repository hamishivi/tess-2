import json
import pdb

import numpy as np


def print_values(path, prefix="logits"):
    name_to_short = {
        "perplexity": "PPL",
        "dist-1": "Dist-1",
        "dist-2": "Dist-2",
        "dist-3": "Dist-3",
        "muave": "MAUVE",
        "repetition": "Repetition",
        "zipf_minus_a": "ZIPF-a",
        "zipf_minus_r": "ZIPF-r",
        "zipf_p": "ZIPF-p",
    }

    if prefix == "logits":
        prefix = "pred_texts_from_logits_masked_"
    else:
        prefix = "pred_texts_from_simplex_masked_"
    ordered_key = ["MAUVE", "PPL", "Dist-1", "Dist-2", "Dist-3", "ZIPF-a", "Repetition"]
    metrics = json.load(open(f"{path}"))
    results = {}
    for k, v in metrics.items():
        for x in name_to_short.keys():
            results[name_to_short[x]] = (
                np.round(100 * metrics[prefix + x], 2)
                if not x
                in [
                    "perplexity",
                    "zipf_minus_a",
                    "repetition",
                ]  # ["PPL", "ZIPF-a", "Repetition"]
                else np.round(metrics[prefix + x], 2)
            )
    values = []
    for k in ordered_key:
        values.append(results[k])
    values = [str(v) for v in values]
    values = "&".join(values)
    print(ordered_key)
    print(values)
    print("_" * 100)


def read_ours():
    path = "/net/nfs.cirrascale/s2-research/rabeehk/outputs/paper_experiments/ours_eval/self_condition_logits_addition_guidance_5_0.99_56_1000/eval_results.json"
    print_values(path)


def read_ours_temperatures_top_p():
    for temperature in [1]:  # [1, 2, 4]:
        for top_p in [0.9, 0.95, 0.99]:  # 0.8,
            print(f"Top-p {top_p} temperature {temperature}")
            # path = f"/net/nfs.cirrascale/s2-research/rabeehk/outputs/paper_experiments/self_condition/tune_length_25_context_25_truncation_56/ul2_self_condition_mean_top_p_{top_p}_temperature_1/checkpoint-13000/eval_results.json"
            # path = f"/net/nfs.cirrascale/s2-research/rabeehk/outputs/paper_experiments/self_condition/tune_length_25_context_25_truncation_56/ul2_self_condition_with_max_top_p_{top_p}_temperature_1/checkpoint-13000/eval_results.json"
            path = f"/net/nfs.cirrascale/s2-research/rabeehk/outputs/paper_experiments/self_condition/tune_length_25_context_25_truncation_56/ul2_self_condition_with_addition_top_p_{top_p}_temperature_1/checkpoint-15000/eval_results.json"
            # /net/nfs.cirrascale/s2-research/rabeehk/outputs/paper_experiments/tune_length_25_context_25_truncation_206/ul2_self_condition_with_addition_top_p_{top_p}_temperature_{temperature}/checkpoint-28000/eval_results.json"
            print_values(path, prefix="logits")


read_ours_temperatures_top_p()
