import json
import os
import pdb

import numpy as np

# path = "/net/nfs.cirrascale/s2-research/rabeehk/outputs/paper_experiments/gpt2_evals_200/"
path = (
    "/net/nfs.cirrascale/s2-research/rabeehk/outputs/paper_experiments/gpt2_evals_256/"
)

name_to_short = {
    "generated_texts_masked_perplexity": "PPL",
    "generated_texts_masked_dist-1": "Dist-1",
    "generated_texts_masked_dist-2": "Dist-2",
    "generated_texts_masked_dist-3": "Dist-3",
    "generated_texts_masked_muave": "MAUVE",
    "generated_texts_masked_repetition": "Repetition",
    "generated_texts_masked_zipf_minus_a": "ZIPF-a",
    "generated_texts_masked_zipf_minus_r": "ZIPF-r",
    "generated_texts_masked_zipf_p": "ZIPF-p",
}

ordered_key = ["MAUVE", "PPL", "Dist-1", "Dist-2", "Dist-3", "ZIPF-a", "Repetition"]
for name in ["gpt2_large_top_p", "gpt2_xl_top_p", "gpt2_medium_top_p"]:
    for top_p in [0.95, 0.99, 0.9]:
        print(f"{name}_{top_p}")
        json_filepath = f"{path}/{name}_{top_p}/metrics.json"
        if os.path.isfile(json_filepath):
            metrics = json.load(open(json_filepath))
        else:
            metrics = np.load(
                f"{path}/{name}_{top_p}/metrics.npy", allow_pickle=True
            ).item()

        metrics = {
            name_to_short[k]: np.round(100 * v, 2)
            if not name_to_short[k] in ["PPL", "ZIPF-a", "Repetition"]
            else np.round(v, 2)
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
