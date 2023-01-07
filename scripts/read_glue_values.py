import numpy as np 
import os 
import json 

# Reads glue values from a folder and computes the average.
output_dir = "/net/nfs.cirrascale/s2-research/rabeehk/outputs/simplex_new/glue_roberta_large_baseline_tuned/"

task_to_metric = {
    "cola": ["matthews_correlation"],
    "mnli": ["accuracy"],
    "mrpc": [ "accuracy", "f1", "combined_score"],
    "qnli": ["accuracy"],
    "qqp": ["accuracy", "f1", "combined_score"],
    "rte": ["accuracy"],
    "sst2": ["accuracy"],
    "stsb": ["pearson", "spearmanr", "combined_score"],
    "wnli": ["accuracy"]
}

glue_ordered = ["cola", "sst2", "mrpc", "qqp", "stsb", "mnli", "qnli", "rte", "wnli"]


results = {}
for task in task_to_metric:
    results[task] = {}
    path = os.path.join(output_dir, task, "test_results.json")
    data = json.load(open(path))
    for metric in task_to_metric[task]:
        results[task][metric] = np.round(data["eval_"+metric]*100, 2)
print(results)

# Computes average.
all_scores = []
for task in results:
    metric = task_to_metric[task][0] if len(task_to_metric[task])==1 else "combined_score"
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