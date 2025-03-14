'''
Evals to plug into the trainer script.
You can always run these directly by running the trainer script 
without the train flag.
'''
import logging
import re
import string
import os
import numpy as np
import pandas as pd

import alpaca_eval
import collections
from datasets import load_dataset, Dataset

from sdlm.inference.inference_utils import process_text
from sdlm.utils import encode_with_messages_format_v1
from sdlm.data.instruction_evals.gsm_exemplars import EXEMPLARS as GSM_EXEMPLARS
from sdlm.data.instruction_evals.codex_evaluation import evaluate_functional_correctness, write_jsonl
from sdlm.data.instruction_evals.squad_eval_1 import evaluate as squad_evaluate
from sdlm.data.instruction_evals.hf_exact_match import exact_match_hf_evaluate as exact_match
from sdlm.data.instruction_evals.ifeval import test_instruction_following_strict, test_instruction_following_loose, load_ifeval_prompts
from sdlm.data.instruction_evals.mmlu_utils import categories as mmlu_categories, subcategories as mmlu_subcategories

logger = logging.getLogger(__name__)

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
        em_score = exact_match(
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
        for example in GSM_EXEMPLARS:
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
            k=[1, 10, 20],
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


subsets = [
        'boolean_expressions', 'causal_judgement', 'date_understanding',
        'disambiguation_qa', 'dyck_languages', 'formal_fallacies', 'geometric_shapes',
        'hyperbaton', 'logical_deduction_five_objects', 'logical_deduction_seven_objects',
        'logical_deduction_three_objects', 'movie_recommendation', 'multistep_arithmetic_two',
        'navigate', 'object_counting', 'penguins_in_a_table', 'reasoning_about_colored_objects',
        'ruin_names', 'salient_translation_error_detection', 'snarks', 'sports_understanding',
        'temporal_sequences', 'tracking_shuffled_objects_five_objects',
        'tracking_shuffled_objects_seven_objects', 'tracking_shuffled_objects_three_objects',
        'web_of_lies', 'word_sorting'
    ]

class BBHEval():
    def compute_metrics(results, skip_special_tokens=True):
        # grab the instructions from the prefixes key
        eval_data = [
            x.replace("<|user|>\n", "").replace("<|assistant|>\nAnswer:", "").strip() for x in results["prefixes"]
        ]
        # for each instruction, grab just the final question
        eval_data = [x.split("\n\nQ:" )[-1].replace("A:", "").strip() for x in eval_data]
        question_to_answer = {}
        for subset in subsets:
            original_data = load_dataset("lukaemon/bbh", subset, split="test")
            for example in original_data:
                question_to_answer[example["input"]] = example["target"]
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
            extracted_answer = re.search(r"[t|T]he answer is (.*?)\.", output)
            if extracted_answer:
                predictions.append(extracted_answer.group(1).strip())
            else:
                predictions.append(output.strip())
        metrics = {}
        # filter out empty gold texts and their corresponding eval data
        predictions = [x for x, y in zip(predictions, gold_texts) if y]
        gold_texts = [x for x in gold_texts if x]
        # now calculate the metrics
        em_score = exact_match(
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
    
    def construct_eval_dataset(tokenizer, max_target_length, max_eval_samples=500):
        # construct prompts
        subset_to_prompt = {}
        for subset in subsets:
            prompt_filename = f"sdlm/data/instruction_evals/bbh-cot-prompts/{subset}.txt"
            with open(prompt_filename, "r") as f:
                task_prompt = "".join(f.readlines()[2:])
            subset_to_prompt[subset] = task_prompt
        prompts = []
        # load the actual samples
        for subset in subsets:
            dataset = load_dataset("lukaemon/bbh", subset, split="test")
            dataset = dataset.shuffle(42)
            if len(dataset) > max_eval_samples:
                dataset = dataset.select(range(max_eval_samples))
            for example in dataset:
                prompt = task_prompt.strip() + "\n\nQ: " + example["input"]
                messages = [{"role": "user", "content": prompt}]
                prompt = encode_with_messages_format_v1(
                    {"messages": messages}, tokenizer, max_target_length, return_string=True
                )
                prompt += "A:" if prompt[-1] in ["\n", " "] else " A:"
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
        eval_dataset = eval_dataset.filter(
            lambda x: any([y != -100 for y in x["labels"]])
        )
        return eval_dataset


squad_shots = [
    "Architecturally, the school has a Catholic character. Atop the Main Building's gold dome is a golden statue of the Virgin Mary. Immediately in front of the Main Building and facing it, is a copper statue of Christ with arms upraised with the legend \"Venite Ad Me Omnes\". Next to the Main Building is the Basilica of the Sacred Heart. Immediately behind the basilica is the Grotto, a Marian place of prayer and reflection. It is a replica of the grotto at Lourdes, France where the Virgin Mary reputedly appeared to Saint Bernadette Soubirous in 1858. At the end of the main drive (and in a direct line that connects through 3 statues and the Gold Dome), is a simple, modern stone statue of Mary.\n\nTo whom did the Virgin Mary allegedly appear in 1858 in Lourdes France?\n\nSaint Bernadette Soubirous",
    "Burke was born in Dublin, Ireland. His mother Mary née Nagle (c. 1702 – 1770) was a Roman Catholic who hailed from a déclassé County Cork family (and a cousin of Nano Nagle), whereas his father, a successful solicitor, Richard (died 1761), was a member of the Church of Ireland; it remains unclear whether this is the same Richard Burke who converted from Catholicism. The Burke dynasty descends from an Anglo-Norman knight surnamed de Burgh (latinised as de Burgo) who arrived in Ireland in 1185 following Henry II of England's 1171 invasion of Ireland.\n\nWhere was Burke born?\n\nDublin, Ireland",
    "The term high definition once described a series of television systems originating from August 1936; however, these systems were only high definition when compared to earlier systems that were based on mechanical systems with as few as 30 lines of resolution. The ongoing competition between companies and nations to create true \"HDTV\" spanned the entire 20th century, as each new system became more HD than the last.In the beginning of the 21st century, this race has continued with 4k, 5k and current 8K systems.\n\nThe term \"high definition\" originally described televisions systems from what year?\n\n1936"
]

class SquadEval():
    def compute_metrics(results, skip_special_tokens=True):
        # grab the instructions from the prefixes key
        eval_data = [
            x.replace("<|user|>\n", "").replace("<|assistant|>\nAnswer:", "").strip() for x in results["prefixes"]
        ]
        # for each, remove the few-shot prompt
        eval_data = [x.replace("\n".join(squad_shots) + '\n', "") for x in eval_data]
        sample_to_answer = {}
        question_to_id = {}
        original_data = load_dataset("squad", split="validation")
        for example in original_data:
            sample_to_answer[example["context"] + "\n\n" + example["question"]] = example
            question_to_id[example["context"] + "\n\n" + example["question"]] = example["id"]
        # final, get ground truth by matching the question
        gold_texts = [sample_to_answer.get(x, None) for x in eval_data]
        ids = [question_to_id.get(x, "") for x in eval_data]
        # then grab from logits masked.
        decoded_preds = (
            process_text(results["pred_texts_from_logits_masked"])
            if not skip_special_tokens
            else results["pred_texts_from_logits_masked"]
        )
        metrics = {}
        # filter out empty gold texts and their corresponding eval data
        predictions = [{"id": y['id'], "prediction_text": x} for x, y in zip(decoded_preds, gold_texts) if y is not None]
        references = [{"id": x["id"], "answers": x["answers"]}  for x in gold_texts if x is not None]
        # now calculate the metrics
        results = squad_evaluate(references=references, predictions=predictions)
        logger.info(f"Results: {results}")
        metrics.update(results)
        return metrics
        
    def construct_eval_dataset(tokenizer, max_target_length, max_eval_samples=500):
        # load the actual samples
        dataset = load_dataset("squad", split="validation")
        dataset = dataset.shuffle(42).select(range(max_eval_samples))
        # convert everything to tulu
        prompts = []
        for sample in dataset:
            prompt = "\n".join(squad_shots) + '\n' + sample["context"] + "\n\n" + sample["question"]
            messages = [{"role": "user", "content": prompt}]
            prompt = encode_with_messages_format_v1(
                {"messages": messages}, tokenizer, max_target_length, return_string=True
            )
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
        eval_dataset = eval_dataset.filter(
            lambda x: any([y != -100 for y in x["labels"]])
        )
        return eval_dataset
    
triviaqa_shots = [
    "Which American-born Sinclair won the Nobel Prize for Literature in 1930?\n\n(Harry) Sinclair Lewis",
    "Where in England was Dame Judi Dench born?\n\nYork, England",
]

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


class TriviaQAEval():
    def compute_metrics(results, skip_special_tokens=True):
        # grab the instructions from the prefixes key
        eval_data = [
            x.replace("<|user|>\n", "").replace("<|assistant|>\nAnswer:", "").strip() for x in results["prefixes"]
        ]
        # for each, remove the few-shot prompt
        eval_data = [x.replace("\n".join(triviaqa_shots) + '\n', "").strip() for x in eval_data]
        sample_to_answer = {}
        question_to_id = {}
        original_data = load_dataset("mandarjoshi/trivia_qa", "rc", split='validation')
        for example in original_data:
            example["id"] = example["question_id"]
            sample_to_answer[example["question"]] = example
            question_to_id[ example["question"]] = example["id"]
        # final, get ground truth by matching the question
        gold_texts = [sample_to_answer.get(x, None) for x in eval_data]
        ids = [question_to_id.get(x, "") for x in eval_data]
        # then grab from logits masked.
        decoded_preds = (
            process_text(results["pred_texts_from_logits_masked"])
            if not skip_special_tokens
            else results["pred_texts_from_logits_masked"]
        )
        metrics = {}
        # filter out empty gold texts and their corresponding eval data
        predictions = [{"id": y['id'], "prediction_text": x} for x, y in zip(decoded_preds, gold_texts) if y is not None]
        references = [{"id": x["id"], "answers": {'text': x["answer"]["aliases"]}}  for x in gold_texts if x is not None]
        # now calculate the metrics
        results = squad_evaluate(references=references, predictions=predictions)
        # also do diffullama-style
        cor = 0
        for pred, ref in zip(predictions, references):
            pred = pred['prediction_text']
            ref = ref['answers']['text']
            for ans in ref:
                if normalize_answer(ans) in normalize_answer(pred.strip()):
                    cor += 1
                    break
        diffullama_acc = cor / len(predictions)
        results["diffullama_acc"] = diffullama_acc
        logger.info(f"Results: {results}")
        metrics.update(results)
        return metrics
        
    def construct_eval_dataset(tokenizer, max_target_length, max_eval_samples=500):
        # load the actual samples
        dataset = load_dataset("mandarjoshi/trivia_qa", "rc", split='validation')
        dataset = dataset.shuffle(42).select(range(max_eval_samples))
        # convert everything to tulu
        prompts = []
        for sample in dataset:
            prompt = "\n".join(triviaqa_shots) + "\n\n" + sample["question"]
            messages = [{"role": "user", "content": prompt}]
            prompt = encode_with_messages_format_v1(
                {"messages": messages}, tokenizer, max_target_length, return_string=True
            )
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
        eval_dataset = eval_dataset.filter(
            lambda x: any([y != -100 for y in x["labels"]])
        )
        return eval_dataset
    

def calculate_scores(outputs):
    """Helper function to calculate accuracy scores from outputs.
    
    Args:
        outputs (list): List of OutputExample objects
        
    Returns:
        dict: Dictionary containing accuracy metrics
    """
    prompt_total = len(outputs)
    prompt_correct = sum(1 for o in outputs if o.follow_all_instructions)
    
    instruction_total = sum(len(o.instruction_id_list) for o in outputs)
    instruction_correct = sum(sum(o.follow_instruction_list) for o in outputs)

    # Calculate per-instruction accuracies
    instruction_metrics = collections.defaultdict(lambda: {"total": 0, "correct": 0})
    
    for output in outputs:
        for inst_id, followed in zip(output.instruction_id_list, output.follow_instruction_list):
            instruction_metrics[inst_id]["total"] += 1
            if followed:
                instruction_metrics[inst_id]["correct"] += 1

    return {
        "prompt_level_accuracy": prompt_correct / prompt_total,
        "instruction_level_accuracy": instruction_correct / instruction_total,
        "per_instruction_accuracy": {
            k: v["correct"] / v["total"] 
            for k, v in instruction_metrics.items()
        }
    }

class IFEval():
    def compute_metrics(results, skip_special_tokens=True):
        import nltk
        nltk.download('punkt')
        nltk.download('punkt_tab')
        """Computes metrics for instruction following evaluation.
        
        Args:
            results (dict): Contains prediction results including prefixes and predictions
            skip_special_tokens (bool): Whether to skip special tokens in processing
        
        Returns:
            dict: Dictionary containing computed metrics
        """
        # Create mapping from prompts to predictions
        prompts = [
            x.replace("<|user|>\n", "").replace("<|assistant|>\n", "").strip() 
            for x in results["prefixes"]
        ]
        
        predictions = (
            process_text(results["pred_texts_from_logits_masked"])
            if not skip_special_tokens
            else results["pred_texts_from_logits_masked"]
        )
        
        # Create prompt -> prediction mapping
        prompt_to_response = {
            prompt: pred.strip() for prompt, pred in zip(prompts, predictions)
        }

        # Read input data
        input_data = load_ifeval_prompts()
        
        metrics = {}
        strict_outputs = []
        loose_outputs = []

        # Process each input example
        for inp in input_data:
            # Skip if prompt not found in predictions
            if inp.prompt not in prompt_to_response:
                continue
                
            # Test instruction following in strict mode
            strict_result = test_instruction_following_strict(
                inp,
                prompt_to_response
            )
            strict_outputs.append(strict_result)

            # Test instruction following in loose mode 
            loose_result = test_instruction_following_loose(
                inp,
                prompt_to_response
            )
            loose_outputs.append(loose_result)

        # Calculate metrics for both strict and loose evaluation
        metrics = {}
        strict_metrics = calculate_scores(strict_outputs)
        for k, v in strict_metrics.items():
            metrics[f"strict_{k}"] = v
        loose_metrics = calculate_scores(loose_outputs)
        for k, v in loose_metrics.items():
            metrics[f"loose_{k}"] = v
        print(metrics)
        return metrics

    def construct_eval_dataset(tokenizer, max_target_length, max_eval_samples=None):
        """Constructs evaluation dataset for instruction following.
        
        Args:
            tokenizer: The tokenizer to use
            max_target_length (int): Maximum sequence length
            max_eval_samples (int): Maximum number of samples to evaluate
        
        Returns:
            Dataset: The constructed evaluation dataset
        """
        # Read the input examples
        eval_dataset = load_ifeval_prompts()

        if max_eval_samples:
            eval_dataset = eval_dataset[:max_eval_samples]

        # Format prompts
        prompts = []
        for sample in eval_dataset:
            messages = [{"role": "user", "content": sample.prompt}]
            prompt = encode_with_messages_format_v1(
                {"messages": messages}, 
                tokenizer,
                max_target_length,
                return_string=True,
                add_generation_prompt=True
            )
            prompts.append(prompt)

        # Tokenize the prompts
        data = tokenizer(
            prompts,
            return_tensors="pt", 
            padding=True,
            truncation=True,
            max_length=max_target_length,
            add_special_tokens=False,
        )
        
        # Convert to Dataset format
        eval_dataset = Dataset.from_dict(data)
        
        # Create labels (-100 for input tokens, 1 for generation space)
        labels = []
        for sample in eval_dataset["input_ids"]:
            labels.append(
                [-100 if x != tokenizer.pad_token_id else 1 for x in sample]
            )
        eval_dataset = eval_dataset.add_column("labels", labels)

        # Filter samples that don't have space for generation
        eval_dataset = eval_dataset.filter(
            lambda x: any([y != -100 for y in x["labels"]])
        )

        return eval_dataset

def calculate_scores(outputs):
    """Helper function to calculate accuracy scores from outputs.
    
    Args:
        outputs (list): List of OutputExample objects
        
    Returns:
        dict: Dictionary containing accuracy metrics
    """
    if not outputs:
        return {
            "prompt_level_accuracy": 0.0,
            "instruction_level_accuracy": 0.0,
            "per_instruction_accuracy": {}
        }
        
    prompt_total = len(outputs)
    prompt_correct = sum(1 for o in outputs if o.follow_all_instructions)
    
    instruction_total = sum(len(o.instruction_id_list) for o in outputs)
    instruction_correct = sum(sum(o.follow_instruction_list) for o in outputs)

    # Calculate per-instruction accuracies
    instruction_metrics = collections.defaultdict(lambda: {"total": 0, "correct": 0})
    
    for output in outputs:
        for inst_id, followed in zip(output.instruction_id_list, output.follow_instruction_list):
            instruction_metrics[inst_id]["total"] += 1
            if followed:
                instruction_metrics[inst_id]["correct"] += 1

    return {
        "prompt_level_accuracy": prompt_correct / prompt_total,
        "instruction_level_accuracy": instruction_correct / instruction_total,
        "per_instruction_accuracy": {
            k: v["correct"] / v["total"] 
            for k, v in instruction_metrics.items()
        }
    }


choices = ["A", "B", "C", "D"]


def format_subject(subject):
    l = subject.split("_")
    s = ""
    for entry in l:
        s += " " + entry
    return s


def format_example(df, idx, include_answer=True):
    prompt = df.iloc[idx, 0]
    k = df.shape[1] - 2
    for j in range(k):
        prompt += "\n{}. {}".format(choices[j], df.iloc[idx, j + 1])
    prompt += "\nAnswer:"
    if include_answer:
        prompt += " {}\n\n".format(df.iloc[idx, k + 1])
    return prompt


def gen_prompt(train_df, subject, k=-1):
    prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(
        format_subject(subject)
    )
    if k == -1:
        k = train_df.shape[0]
    for i in range(k):
        prompt += format_example(train_df, i)
    return prompt

class MMLUEval():
    def compute_metrics(results, skip_special_tokens=True):
        # Extract prefixes and predictions
        eval_data = [
            x.replace("<|user|>\n", "").replace("<|assistant|>\nAnswer:", "").strip() 
            for x in results["prefixes"]
        ]
        
        # Get predictions
        decoded_preds = (
            process_text(results["pred_texts_from_logits_masked"])
            if not skip_special_tokens
            else results["pred_texts_from_logits_masked"]
        )
        
        # Process each prediction to extract the answer choice
        predictions = []
        choices = ["A", "B", "C", "D"]
        for pred in decoded_preds:
            pred = pred.strip()
            # Take first character if it's a valid choice
            if pred and pred[0] in choices:
                predictions.append(pred[0])
            else:
                # Default to first choice if invalid
                predictions.append(choices[0])
                
        # Calculate metrics for each category and subcategory
        all_cors = []
        subcat_cors = {
            subcat: [] for subcat_lists in mmlu_subcategories.values() for subcat in subcat_lists
        }
        cat_cors = {cat: [] for cat in mmlu_categories}
        
        # Match questions to answers from the dataset
        for i, prompt in enumerate(eval_data):
            # Extract subject from prompt format
            subject = prompt.split("The following are multiple choice questions (with answers) about")[1].split(".")[0].strip()

            subject = subject.replace(" ", "_")
            
            # Load test data for this subject
            test_df = pd.read_csv(
                os.path.join("/weka/oe-adapt-default/hamishi/simplex-diffusion/sdlm/data/instruction_evals/mmlu_data/data/test", f"{subject}_test.csv"),
                header=None
            )
            
            # Get ground truth
            ground_truth = test_df.iloc[i % len(test_df), -1]
            correct = predictions[i] == ground_truth
            
            # Update metrics
            all_cors.append(correct)
            
            # Update category metrics
            subcats = mmlu_subcategories[subject]
            for subcat in subcats:
                subcat_cors[subcat].append(correct)
                for key in mmlu_categories.keys():
                    if subcat in mmlu_categories[key]:
                        cat_cors[key].append(correct)
                        
        # Calculate final metrics
        metrics = {
            "average_acc": np.mean(all_cors),
        }
        for subcat in subcat_cors:
            metrics[f"{subcat}_acc"] = np.mean(subcat_cors[subcat])
        for cat in cat_cors:
            metrics[f"{cat}_acc"] = np.mean(cat_cors[cat])
        
        return metrics
        
    def construct_eval_dataset(tokenizer, max_target_length, max_eval_samples=None):
        # Get list of subjects
        subjects = sorted([
            f.split("_test.csv")[0]
            for f in os.listdir(os.path.join("/weka/oe-adapt-default/hamishi/simplex-diffusion/sdlm/data/instruction_evals/mmlu_data/data/test"))
            if "_test.csv" in f
        ])
        
        if max_eval_samples:
            subjects = subjects[:max_eval_samples]
            
        prompts = []
        for subject in subjects:
            # Load dev and test data
            dev_df = pd.read_csv(
                os.path.join("/weka/oe-adapt-default/hamishi/simplex-diffusion/sdlm/data/instruction_evals/mmlu_data/data/dev", f"{subject}_dev.csv"),
                header=None
            )
            test_df = pd.read_csv(
                os.path.join("/weka/oe-adapt-default/hamishi/simplex-diffusion/sdlm/data/instruction_evals/mmlu_data/data/test", f"{subject}_test.csv"),
                header=None
            )
            
            # Format prompts with few-shot examples
            for i in range(len(test_df)):
                k = 0  # Number of few-shot examples
                prompt_end = format_example(test_df, i, include_answer=False)
                train_prompt = gen_prompt(dev_df, subject, k)
                prompt = train_prompt + prompt_end
                
                messages = [{"role": "user", "content": prompt}]
                formatted_prompt = encode_with_messages_format_v1(
                    {"messages": messages},
                    tokenizer,
                    max_target_length,
                    return_string=True,
                    add_generation_prompt=True
                )
                prompts.append(formatted_prompt)
                
        # Tokenize all prompts
        data = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_target_length,
            add_special_tokens=False,
        )
        
        eval_dataset = Dataset.from_dict(data)
        
        # Create labels (-100 for input tokens, 1 for generation space)
        labels = []
        for sample in eval_dataset["input_ids"]:
            if tokenizer.pad_token_id not in sample:
                labels.append([-100 for _ in sample])
                continue
            first_pad_idx = sample.index(tokenizer.pad_token_id)
            second_pad_idx = first_pad_idx + 1
            # if too long, just continue, we will filter out.
            if second_pad_idx >= len(sample):
                labels.append([-100 for _ in sample])
                continue
            label = [-100 for _ in sample]
            # MMLU difference: only leave space for answer + eos token
            label[first_pad_idx] = 1
            label[second_pad_idx] = 1
            labels.append(label)
        eval_dataset = eval_dataset.add_column("labels", labels)
        
        # Filter samples without generation space
        eval_dataset = eval_dataset.filter(
            lambda x: any([y != -100 for y in x["labels"]])
        )
        
        return eval_dataset


EVAL_MAPPING = {
    "alpaca_eval": AlpacaEval,
    "gsm8k": GSM8kEval,
    "human_eval": CodexHumanEval,
    "bbh": BBHEval,
    "squad": SquadEval,
    "triviaqa": TriviaQAEval,
    "ifeval": IFEval,
    "mmlu": MMLUEval
}
