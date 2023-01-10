# computes the metrics.
import sys
import os
from transformers import HfArgumentParser
from sdlm.arguments import DataTrainingArguments, ModelArguments, TrainingArguments, DiffusionArguments
from run_mlm import get_compute_metrics
from sdlm.trainer import GENERATION_RESULTS
import json
from transformers.trainer_utils import denumpify_detensorize


def save_metrics(metrics, split, training_args):
    path = os.path.join(training_args.output_dir, f"{split}_results.json")
    with open(path, "w") as f:
        json.dump(metrics, f, indent=4, sort_keys=True)


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments, DiffusionArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        # NOTE: keep the diffusion_args to avoid the error when we call the scripts with all the arguments
        # used in running `run_mlm.py`.
        model_args, data_args, training_args, diffusion_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args, diffusion_args = parser.parse_args_into_dataclasses()

    compute_metrics = get_compute_metrics(data_args, training_args, model_args)
    # Load generations.
    results = json.load(open(os.path.join(training_args.output_dir, GENERATION_RESULTS + "_eval_results.json")))
    metrics = compute_metrics(results)
    metrics = denumpify_detensorize(metrics)
    save_metrics(metrics, "eval", training_args)


if __name__ == "__main__":
    main()
