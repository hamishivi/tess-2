import json
import pdb

import numpy as np

# path = "/net/nfs.cirrascale/s2-research/rabeehk/outputs/paper_experiments/gold_text_with_gpt_large_256"
path = "/net/nfs.cirrascale/s2-research/rabeehk/outputs/paper_experiments/gold_truncation__context_size_25"

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
metrics = json.load(open(f"{path}/metrics.json"))
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
