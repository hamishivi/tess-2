'''
Evals to plug into the trainer script.
You can always run these directly by running the trainer script 
without the train flag.
'''
import logging
import re
import json

import alpaca_eval
import evaluate
from datasets import load_dataset, Dataset

from sdlm.inference.inference_utils import process_text
from sdlm.utils import encode_with_messages_format_v1
from sdlm.data.instruction_evals.gsm_exemplars import EXEMPLARS as GSM_EXEMPLARS
from sdlm.data.instruction_evals.codex_evaluation import evaluate_functional_correctness, write_jsonl

logger = logging.getLogger(__name__)

exact_match = evaluate.load("exact_match")

class DiffusionEvaluation():
    def compute_metrics(results, skip_special_tokens=True):
        pass

    def construct_eval_dataset(tokenizer, max_target_length, max_eval_samples=None):
        pass

class AlpacaEval():
    def compute_metrics(results, skip_special_tokens=True):
        # grab the instructions from the prefixes key
        eval_data = [
            x.replace("<|user|>\n", "").replace("<|assistant|>\n", "").strip() for x in results["prefixes"]
        ]
        # then grab from logits masked.
        decoded_preds = (
            process_text(results["pred_texts_from_logits_masked"])
            if not skip_special_tokens
            else results["pred_texts_from_logits_masked"]
        )
        decoded_preds = [x.strip() for x in decoded_preds]
        metrics = {}
        # for each decoded sample, format into alpacaeval setup
        decoded_preds = [
            {"output": y, "instruction": x, "generator": "tess2"}
            for x, y in zip(eval_data, decoded_preds)
        ]
        # sometimes in multi-process envs we get a few extra samples.
        if len(decoded_preds) > 805:
            # keep only unique instructions
            unique_instructions = set()
            unique_preds = []
            for pred in decoded_preds:
                if pred["instruction"] not in unique_instructions:
                    unique_instructions.add(pred["instruction"])
                    unique_preds.append(pred)
            decoded_preds = unique_preds
        
        df_leaderboard, _ = alpaca_eval.evaluate(
            model_outputs=decoded_preds,
            is_overwrite_leaderboard=True,
            is_return_instead_of_print=True,
        )
        # grab tess2 results
        key_metrics = df_leaderboard.loc["tess2"].to_dict()
        metrics.update(key_metrics)
        return metrics

    def construct_eval_dataset(tokenizer, max_target_length, max_eval_samples=None):
        logger.warn(
            "Running evaluation. This calls GPT-4, so PLEASE MAKE SURE YOU ARE NOT RUNNING IT A TONNE"
        )
        eval_dataset = load_dataset("tatsu-lab/alpaca_eval")["eval"]
        # put the dataset into the correct format
        eval_dataset = eval_dataset.map(
            lambda x: {"messages": [{"role": "user", "content": x["instruction"]}]}
        )
        if max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))
        tokenized_data = []
        for sample in eval_dataset:
            prompt = encode_with_messages_format_v1(
                sample, tokenizer, max_target_length, return_string=True
            )
            prompt = prompt + "\n<|assistant|>\n"
            tokenized_data.append(prompt)
        data = tokenizer(
            tokenized_data,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_target_length,
            add_special_tokens=False,
        )
        eval_dataset = Dataset.from_dict(data)
        labels = []
        # we dont assume a length on the response.
        # so labels are -100 for for inputs, and 1 everywhere else.
        # eval loss is meaningless here.
        for sample in eval_dataset["input_ids"]:
            labels.append(
                [-100 if x != tokenizer.pad_token_id else 1 for x in sample]
            )
        eval_dataset = eval_dataset.add_column("labels", labels)
        # filter out samples without any space for generations.
        # for roberta (512), should just be one.
        eval_dataset = eval_dataset.filter(
            lambda x: any([y != -100 for y in x["labels"]])
        )
        return eval_dataset
    
class GSM8kEval():
    def compute_metrics(results, skip_special_tokens=True):
        # grab the instructions from the prefixes key
        eval_data = [
            x.replace("<|user|>\n", "").replace("<|assistant|>\nAnswer:", "").strip() for x in results["prefixes"]
        ]
        # for each instruction, grab just the final question
        eval_data = [x.split("Question: ")[-1].strip() for x in eval_data]
        original_data = load_dataset("openai/gsm8k", "main", split="test")
        question_to_answer = {}
        for example in original_data:
            answer = example["answer"].split("####")[1].strip()
            answer = re.sub(r"(\d),(\d)", r"\1\2",answer)
            question_to_answer[example["question"]] = answer
        # final, get ground truth by matching the question
        gold_texts = [question_to_answer.get(x, "") for x in eval_data]
        # then grab from logits masked.
        decoded_preds = (
            process_text(results["pred_texts_from_logits_masked"])
            if not skip_special_tokens
            else results["pred_texts_from_logits_masked"]
        )
        predictions = []
        for output in decoded_preds:
            # replace numbers like `x,xxx` with `xxxx`
            output = re.sub(r"(\d),(\d)", r"\1\2", output)
            numbers = re.findall(r"[-+]?\d*\.\d+|\d+", output)
            if numbers:
                predictions.append(numbers[-1])
            else:
                predictions.append(output)
        metrics = {}
        # filter out empty gold texts and their corresponding eval data
        predictions = [x for x, y in zip(predictions, gold_texts) if y]
        gold_texts = [x for x in gold_texts if x]
        # now calculate the metrics
        em_score = exact_match.compute(
            predictions=predictions,
            references=gold_texts,
            ignore_case=True,
            ignore_punctuation=True
        )['exact_match']
        logger.info(f"EM: {em_score}")
        # update the metrics
        key_metrics = {"EM": em_score}
        metrics.update(key_metrics)
        return metrics

    def construct_eval_dataset(tokenizer, max_target_length, max_eval_samples=200):
        eval_dataset = load_dataset("openai/gsm8k", "main", split="test")
        max_eval_samples = min(len(eval_dataset), max_eval_samples)
        logger.info(f"We are using {max_eval_samples} samples")
        eval_dataset = eval_dataset.shuffle(42).select(range(max_eval_samples))
        # put the dataset into the correct format
        # for gsm8k, we will use 3-shot cot to match standard setups.
        # why 3-shot? 512 context length means we cant fit 8 
        global GSM_EXEMPLARS
        demonstrations = []
        for example in GSM_EXEMPLARS[:3]:
            demonstrations.append(
                "Question: " + example["question"] + "\n" + "Answer: " + example["cot_answer"]
            )
        prompt_prefix = "Answer the following questions.\n\n" + "\n\n".join(demonstrations) + "\n\n"
        # format out the answers so the answer is number only. this will be our label.
        labels = []
        for example in eval_dataset:
            answer = example["answer"].split("####")[1].strip()
            answer = re.sub(r"(\d),(\d)", r"\1\2",answer)
            assert float(answer), f"answer is not a valid number: {example['answer']}"
            labels.append(answer)
        eval_dataset = eval_dataset.map(
            lambda x: {"messages": [{"role": "user", "content": prompt_prefix + "Question: " + x["question"].strip()}]}
        )
        tokenized_data = []
        tokenized_and_labelled_data = []
        for sample, label in zip(eval_dataset, labels):
            prompt = encode_with_messages_format_v1(
                sample, tokenizer, max_target_length, return_string=True
            )
            prompt = prompt + "\n<|assistant|>\nAnswer:"
            tokenized_data.append(prompt)
            tokenized_and_labelled_data.append(prompt + label + "\n")
        data = tokenizer(
            tokenized_data,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_target_length,
            add_special_tokens=False,
        )
        eval_dataset = Dataset.from_dict(data)
        labelled_data = tokenizer(
            tokenized_and_labelled_data,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_target_length,
            add_special_tokens=False,
        )
        eval_labelled_dataset = Dataset.from_dict(labelled_data)
        # labels are -100 on matching
        labels = []
        # we dont assume a length on the response.
        # so labels are -100 for for inputs, and 1 everywhere else.
        # eval loss is meaningless here.
        for sample in eval_dataset["input_ids"]:
            labels.append(
                [-100 if x != tokenizer.pad_token_id else 1 for x in sample]
            )
        eval_dataset = eval_dataset.add_column("labels", labels)
        # filter out samples without any space for generations.
        # for roberta (512), should just be one.
        eval_dataset = eval_dataset.filter(
            lambda x: any([y != -100 for y in x["labels"]])
        )
        return eval_dataset


class CodexHumanEval():
    def compute_metrics(results, skip_special_tokens=True):
        # grab the instructions from the prefixes key
        eval_data = [
            x.replace("<|user|>", "").replace("<|assistant|>", "").strip() for x in results["prefixes"]
        ]
        # load eval data and match it up
        original_data = load_dataset("openai/openai_humaneval", split="test")
        question_to_answer = {}
        for sample in eval_data:
            for example in original_data:
                if example["prompt"].strip() in sample:
                    question_to_answer[sample] = example
                    break
        # then grab from logits masked.
        decoded_preds = (
            process_text(results["pred_texts_from_logits_masked"])
            if not skip_special_tokens
            else results["pred_texts_from_logits_masked"]
        )
        # process text consistently removes a space from the start, which messes up indentation
        decoded_preds = [" " + x for x in decoded_preds]
        # cut the preds off in the same way we do stop seqs in the AR setting
        stop_sequences = ["\nclass", "\ndef", "\n#", "\nif", "\nprint", "\n```"]
        for i, _ in enumerate(decoded_preds):
            for stop_seq in stop_sequences:
                if stop_seq in decoded_preds[i]:
                    decoded_preds[i] = decoded_preds[i].split(stop_seq)[0]
        # okay, now we can construct our predictions
        predictions = []
        generated_solutions = set()
        for prediction, sample in zip(decoded_preds, eval_data):
            original_sample = question_to_answer[sample]
            predictions.append({
                "task_id": original_sample["task_id"],
                "prompt": original_sample["prompt"],
                "completion": prediction
            })
            generated_solutions.add(original_sample["task_id"])
        # save the predictions - the eval needs this
        prediction_save_path = "codex_human_eval_predictions.jsonl"
        write_jsonl(prediction_save_path, predictions)
        # now calculate the metrics
        # for now, just p@1 since higher is annoying.
        # we could do it in the future.
        # only pass through problems we actually evaluate on.
        metrics = evaluate_functional_correctness(
            sample_file=prediction_save_path,
            k=[1, 10],
            problems={example["task_id"]: example for example in original_data if example["task_id"] in generated_solutions},
            n_workers=64
        )
        logger.info(f"Results: {metrics}")
        return metrics

    def construct_eval_dataset(tokenizer, max_target_length, max_eval_samples=500):
        eval_dataset = load_dataset("openai/openai_humaneval", split="test")
        # use hep for better prompting
        instructions = load_dataset("bigcode/humanevalpack", "python")["test"]
        # only 164 samples, so this probably shouldnt come into play much
        max_eval_samples = min(len(eval_dataset), max_eval_samples)
        logger.info(f"We are using {max_eval_samples} samples")
        eval_dataset = eval_dataset.shuffle(42).select(range(max_eval_samples))
        # put the dataset into the correct format
        # humaneval is 0-shot, but with some prompts, so should be chill.
        instructions_dict = {
            x["task_id"].replace("Python", "HumanEval"): x["instruction"] for x in instructions
        }
        answer = "Here is the function:\n\n```python\n"
        prompts = []
        for example in eval_dataset:
            messages = [{"role": "user", "content": instructions_dict[example["task_id"]]}]
            prompt = encode_with_messages_format_v1(
                {"messages": messages}, tokenizer, max_target_length, return_string=True
            )
            prompt = prompt + "\n<|assistant|>\n" + answer + example["prompt"]
            prompts.append(prompt)
        data = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_target_length,
            add_special_tokens=False,
        )
        eval_dataset = Dataset.from_dict(data)
        # labels are -100 on any non-pad token
        labels = []
        for sample in eval_dataset["input_ids"]:
            labels.append(
                [-100 if x != tokenizer.pad_token_id else 1 for x in sample]
            )
        eval_dataset = eval_dataset.add_column("labels", labels)
        # filter out samples without any space for generations.
        # for roberta (512), should just be one.
        eval_dataset = eval_dataset.filter(
            lambda x: any([y != -100 for y in x["labels"]])
        )
        # finally, duplicate each example 20 times - this is the number of samples we will generate.
        new_eval_dataset = []
        for example in eval_dataset:
            for _ in range(20):
                new_eval_dataset.append(example)
        eval_dataset = Dataset.from_list(new_eval_dataset)
        return eval_dataset

EVAL_MAPPING = {
    "alpaca_eval": AlpacaEval,
    "gsm8k": GSM8kEval,
    "human_eval": CodexHumanEval,
}