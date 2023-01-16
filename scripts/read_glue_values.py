import numpy as np
import os
import json
import pdb

# Reads glue values from a folder and computes the average.

task_to_metric = {
    "cola": ["matthews_correlation"],
    "mnli": ["accuracy"],
    "mrpc": ["accuracy", "f1", "combined_score"],
    "qnli": ["accuracy"],
    "qqp": ["accuracy", "f1", "combined_score"],
    "rte": ["accuracy"],
    "sst2": ["accuracy"],
    "stsb": ["pearson", "spearmanr", "combined_score"],
    "wnli": ["accuracy"],
}

glue_ordered = ["cola", "sst2", "mrpc", "qqp", "stsb", "mnli", "qnli", "rte", "wnli"]


def read_values(paths):
    results = {}
    for task in task_to_metric:
        results[task] = {}
        path = paths[task]
        data = json.load(open(path))
        for metric in task_to_metric[task]:
            results[task][metric] = np.round(data["eval_pred_texts_from_logits_masked_" + metric], 2)
    print(results)

    # Computes average.
    all_scores = []
    for task in results:
        metric = task_to_metric[task][0] if len(task_to_metric[task]) == 1 else "combined_score"
        all_scores.append(results[task][metric])
    avg_results = np.round(np.mean(all_scores), 2)
    print("Average", avg_results)

    # Show results in the format of latex.
    table_row = []
    for task in glue_ordered:
        task_results = []
        for metric in task_to_metric[task]:
            if metric == "combined_score":
                continue
            task_results.append(results[task][metric])
        task_results = [str(t) for t in task_results]
        table_row.append("/".join(task_results))

    table_row.append(str(avg_results))
    print("&".join(table_row))


"""
output_dir = "/net/nfs.cirrascale/s2-research/rabeehk/outputs/simplex_new/glue_roberta_large_baseline_tuned/"
paths={task:os.path.join(output_dir, task, "test_results.json") for task in task_to_metric.keys()}
read_values(paths)
"""

# Read glue values.
output_dir = "/net/nfs.cirrascale/s2-research/rabeehk/outputs/paper_experiments/ours_glue_self_condition_mean"
dirs = {
    "cola": "cola_steps_10_wd_0.01",
    "mnli": "mnli_steps_10_wd_0.01",
    "mrpc": "mrpc_steps_10_wd_0.01",
    "qnli": "qnli_steps_10_wd_0.01",
    "qqp": "qqp_steps_10_wd_0.01",
    "sst2": "sst2_steps_10_wd_0.01",
    "stsb": "stsb_steps_10_wd_0.01",
    "mnli": "mnli_steps_10_wd_0.01",
    "mrpc": "mrpc_steps_10_wd_0.01",
    "rte": "rte_steps_10_wd_0.01",
    "sst2": "sst2_steps_10_wd_0.01",
    "wnli": "wnli_steps_10_wd_0.01",
}
paths = {}
for task in dirs:
    paths[task] = os.path.join(output_dir, dirs[task], "test_results.json")
read_values(paths)
