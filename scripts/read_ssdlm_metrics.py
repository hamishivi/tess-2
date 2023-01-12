import pdb
import json
import numpy as np

#path = "/net/nfs.cirrascale/s2-research/rabeehk/ssd-lm/logging/ssd_dbs25/ctx25_trunc56_depth7_ctrlr0.0_step1000_topp0.9_ssd_gen_sampling_metrics.json"
path="/net/nfs.cirrascale/s2-research/rabeehk/ssdlm-baseline/logging/ssd_dbs25/ctx25_trunc206_depth1_ctrlr0.0_step1000_topp0.99_ssd_gen_sampling_metrics.json"

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
