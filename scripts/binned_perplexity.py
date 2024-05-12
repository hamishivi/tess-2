import argparse
import json

from transformers import AutoModelForCausalLM, AutoTokenizer

from sdlm.metrics.perplexity import conditional_perplexity, perplexity

parser = argparse.ArgumentParser()
parser.add_argument(
    "--predictions", "-p", type=str, required=True
)  # prediction file from training.
parser.add_argument(
    "--stride", "-s", type=int, default=50
)  # stride for bucketing perplexities
args = parser.parse_args()

stride = args.stride

with open(args.predictions, "r") as f:
    data = json.load(f)
predictions = data["inference_100_pred_texts_from_logits_masked"]
prefixes = data["inference_100_prefixes"]


model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B")
model = model.cuda()
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")
original_tokenizer = AutoTokenizer.from_pretrained("roberta-base")

comparison_model = AutoModelForCausalLM.from_pretrained("gpt2").cuda()
comparison_tokenizer = AutoTokenizer.from_pretrained("gpt2")


max_len = len(original_tokenizer(predictions[0]).input_ids)


# generate using gpt-neo-1.3B for comparison.
causal_preds = []
for prefix in prefixes:
    inputs = comparison_tokenizer(prefix, return_tensors="pt").input_ids.cuda()
    causal_preds.append(
        comparison_tokenizer.decode(
            comparison_model.generate(
                input_ids=inputs,
                max_new_tokens=max_len,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )[0][inputs.shape[1] :],
            skip_special_tokens=True,
        )
    )

split_preds = {}
split_prefixes = {}
last_i = 0
for i in range(0, 128, stride):
    split_preds[i] = []
    split_prefixes[i] = []
    for j, pred in enumerate(predictions):
        split = original_tokenizer.decode(
            original_tokenizer(pred).input_ids[i : i + stride], skip_special_tokens=True
        )
        split_preds[i].append(split)
        prefix = prefixes[j]
        if last_i > 0:
            prefix = prefix + " " + split_prefixes[last_i][j]
        split_prefixes[i].append(prefix)

# remove all empty preds
for r in split_preds:
    split_prefixes[r] = [
        prefix
        for j, prefix in enumerate(split_prefixes[r])
        if len(split_preds[r][j]) > 0
    ]
    split_preds[r] = [pred for pred in split_preds[r] if len(pred) > 0]


split_causal_preds = {}
split_causal_prefixes = {}
last_i = 0
for i in range(0, 128, stride):
    split_causal_preds[i] = []
    split_causal_prefixes[i] = []
    for j, pred in enumerate(causal_preds):
        split = original_tokenizer.decode(
            original_tokenizer(pred).input_ids[i : i + stride], skip_special_tokens=True
        )
        split_causal_preds[i].append(split)
        prefix = prefixes[j]
        if last_i > 0:
            prefix = prefix + " " + split_causal_prefixes[last_i][j]
        split_causal_prefixes[i].append(prefix)

# remove all empty preds
for r in split_causal_preds:
    split_causal_prefixes[r] = [
        prefix
        for j, prefix in enumerate(split_causal_prefixes[r])
        if len(split_causal_preds[r][j]) > 0
    ]
    split_causal_preds[r] = [pred for pred in split_causal_preds[r] if len(pred) > 0]


# we split predictions into stride-length tokens
counter = 0
for r in split_preds:
    if len(split_preds[r]) == 0:
        continue
    metrics = conditional_perplexity(
        split_preds[r], split_prefixes[r], model, tokenizer
    )
    print(
        f"Conditional Perplexity for tokens {r}-{min(r+stride, max_len)}: {metrics['mean_perplexity']}"
    )
    counter += r

counter = 0
for r in split_preds:
    if len(split_preds[r]) == 0:
        continue
    metrics = perplexity(split_preds[r], model, tokenizer)
    print(
        f"Unconditional Perplexity for tokens {r}-{min(r+stride, max_len)}: {metrics['mean_perplexity']}"
    )
    counter += r

print("-------------- CAUSAL MODEL ⬇️ -----------------------")

counter = 0
for r in split_causal_preds:
    if len(split_causal_preds[r]) == 0:
        continue
    metrics = conditional_perplexity(
        split_causal_preds[r], split_causal_prefixes[r], model, tokenizer
    )
    print(
        f"Conditional Perplexity for tokens {r}-{min(r+stride, max_len)}: {metrics['mean_perplexity']}"
    )
    counter += r

counter = 0
for r in split_causal_preds:
    if len(split_causal_preds[r]) == 0:
        continue
    metrics = perplexity(split_causal_preds[r], model, tokenizer)
    print(
        f"Unconditional Perplexity for tokens {r}-{min(r+stride, max_len)}: {metrics['mean_perplexity']}"
    )
    counter += r
