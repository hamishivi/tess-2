'''
Script to score rewards with a reward model
'''
import argparse
import json
import torch
from tqdm import tqdm
from sdlm.models.utils import load_classifier

parser = argparse.ArgumentParser()
parser.add_argument('--reward_model', type=str)  # reward model
parser.add_argument('--input', type=str)  # input to score
parser.add_argument('--output', type=str)  # output to score
args = parser.parse_args()

tokenizer, model = load_classifier(args.reward_model)
model = model.cuda()

with open(args.input, 'r') as f:
    inputs = json.load(f)

samples = inputs
data = [f"<|user|>\n{input['instruction']}\n<|assistant|>\n{input['output']}" + tokenizer.eos_token for input in inputs]

# score~!
scores = []
for d in tqdm(data):
    inputs = tokenizer(d, return_tensors='pt', padding=True, truncation=True).input_ids.cuda()
    outputs = model(inputs)
    scores.append(outputs.logits.detach().cpu().to(torch.float32).numpy().item())

with open(args.output, 'w') as f:
    for score, sample in zip(scores, samples):
        f.write(json.dumps({'score': score, **sample}) + '\n')

# print summary statistics
print(f"Mean: {sum(scores) / len(scores)}")
print(f"Max: {max(scores)}")
print(f"Min: {min(scores)}")
print(f"Std: {sum([(score - sum(scores) / len(scores))**2 for score in scores]) / len(scores)}")
print(f"Num samples: {len(scores)}")
