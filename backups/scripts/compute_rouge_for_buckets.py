# buckets the summaries and computes the rouge for each bucket.
import json
import pdb
from collections import Counter

import evaluate
import numpy as np
from data.postprocessors import postprocess_text_for_metric
from transformers import AutoTokenizer


def extract_results(indices, lengths, data, x):
    extracted_indices = []
    for i, value in enumerate(indices):
        if value == x:
            extracted_indices.append(i)
    extracted_data = {}
    for k, v in data.items():
        extracted_data[k] = np.array(v)[extracted_indices]
    return extracted_data


def compute_metrics(results):
    keys = ["pred_texts_from_logits_masked"]
    metrics = {}
    for key in keys:
        decoded_preds = (
            process_text(results[key]) if not skip_special_tokens else results[key]
        )
        # Note that since decoded_labels is getting updated after post-process, we
        # need to compute it here for each key.
        decoded_labels = (
            process_text(results["gold_texts_masked"])
            if not skip_special_tokens
            else results["gold_texts_masked"]
        )
        decoded_preds, decoded_labels = postprocess_text_for_metric(
            "rouge", decoded_preds, decoded_labels
        )
        key_metrics = metric.compute(
            predictions=decoded_preds, references=decoded_labels, use_stemmer=True
        )
        key_metrics = {k: round(v * 100, 4) for k, v in key_metrics.items()}
        key_metrics = {f"{key}_{k}": v for k, v in key_metrics.items()}
        metrics.update(key_metrics)
    return metrics


skip_special_tokens = True
generation_path = "/net/nfs.cirrascale/s2-research/rabeehk/outputs/paper_experiments/cnn_dailymail_results/ours_lr_3e-5_max_steps_120000_model_roberta-base/generated_test_top_p_None_temperature_1.0_seed_42_results.json"
data = json.load(open(generation_path))
tokenizer = AutoTokenizer.from_pretrained("roberta-base")
targets = data["gold_texts_masked"]

lengths = []
for target in targets:
    tokens = tokenizer(target)["input_ids"]
    lengths.append(len(tokens))

pred_lengths = []
for pred in data["pred_texts_from_logits_masked"]:
    tokens = tokenizer(pred)["input_ids"]
    pred_lengths.append(len(tokens))

pdb.set_trace()
bins = np.arange(0, 150, 25)
indices = np.digitize(lengths, bins)
counts = Counter(indices)
print(counts)
metric = evaluate.load("rouge")
unique_indices = np.unique(indices)
bucket_metrics = {}
for x in unique_indices:
    extracted_data = extract_results(indices, lengths, data, x)
    bucket_results = compute_metrics(extracted_data)
    bucket_metrics[str(x)] = bucket_results

bucket_metrics["target_lengths"] = lengths
bucket_metrics["pred_lengths"] = pred_lengths

with open("our_model_cnn_bucket_results.json", "w") as outfile:
    json.dump(bucket_metrics, outfile)
