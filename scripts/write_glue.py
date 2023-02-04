import pdb
import json 
import csv 
import os 

def map_predictions(predictions, dataset):
    label_to_word = {"0": "entailment", "1": "not_entailment"}
    label_to_word_threeway = {"0": "entailment", "1": "neutral", "2": "contradiction"}
    if dataset == "qnli":
        predictions = [label_to_word[prediction] for prediction in predictions]
    elif dataset == "rte":
        predictions = [label_to_word[prediction] for prediction in predictions]
    elif dataset == "mnli":
        predictions = [label_to_word_threeway[prediction] for prediction in predictions]
    return predictions


dataset_to_filename = {"cola": "CoLA", "mrpc": "MRPC", "sst2": "SST-2", "rte": "RTE", "qqp": "QQP", "qnli": "QNLI", "wnli": "WNLI", "stsb":"STS-B", "mnli": "MNLI-m"}
OUTPUT_DIR = "glue_results"
def write_to_a_file(predictions, filename, dataset):
    predictions = map_predictions(predictions, dataset)
    label_file = f"{OUTPUT_DIR}/{filename}.tsv"
    with open(label_file, 'w', encoding='utf8', newline='') as tsv_file:
        tsv_writer = csv.writer(tsv_file, delimiter='\t', lineterminator='\n')
        tsv_writer.writerow(["index", "prediction"])
        for index, label in enumerate(predictions):
            tsv_writer.writerow([index, label])

if not os.path.isdir(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR) 

datasets = ["qnli",    "rte",  "stsb",  "wnli",  "qqp",     "mrpc",    "sst2", "mnli", "cola"]
for dataset in datasets:
    path=f"/net/nfs.cirrascale/s2-research/rabeehk/outputs/paper_experiments/glue_results/ours_self_condition_mean_mix_before_weights_{dataset}_steps_10_no_wd_max_steps_set/all_data_eval/generated_test_top_p_None_temperature_1.0_seed_42_guidance_scale_1.0_results.json"
    f = open(path)
    predictions = json.load(f)["pred_texts_from_logits_masked"]
    filename = dataset_to_filename[dataset]
    write_to_a_file(predictions, filename, dataset)
    pdb.set_trace()


