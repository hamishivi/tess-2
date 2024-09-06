'''
Compute the correlation between the reward and alpacaeval judgements.
'''
import argparse
import json
import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--alpacaeval_annotations', type=str)  # annotations with alpacaeval
parser.add_argument('--reward_results', type=str, nargs='+')  # reward scores on some output
args = parser.parse_args()

# /home/hamishi/.conda/envs/simplex-diffusion/lib/python3.11/site-packages/alpaca_eval/evaluators_configs/alpaca_eval_gpt4/annotations_seed0_configs.json

# load alpacaeval annotations
with open(args.alpacaeval_annotations, 'r') as f:
    alpacaeval_annotations = json.load(f)
print(len(alpacaeval_annotations))
annotation_dict = {}
for annotation in alpacaeval_annotations:
    instruction = annotation['instruction']
    if instruction not in annotation_dict:
        annotation_dict[instruction] = []
    annotation_dict[instruction].append(annotation)
alpacaeval_annotations = annotation_dict

# now, go through the reward results and get annotations
reward_results = []
for reward_result in args.reward_results:
    with open(reward_result, 'r') as f:
        reward_results += [json.loads(line) for line in f]

for result in tqdm(reward_results):
    instruction = result['instruction']
    if instruction in alpacaeval_annotations:
        annotations = alpacaeval_annotations[instruction]
        for annotation in annotations:
            if annotation['output_1'] == result['output'] and annotation['preference'] == 1.0:
                result['alpaca_score'] = 1
                break
            elif annotation['output_2'] == result['output'] and annotation['preference'] == 2.0:
                result['alpaca_score'] = 1
                break
            elif annotation['output_1'] == result['output'] and annotation['preference'] == 2.0:
                result['alpaca_score'] = 0
                break
            elif annotation['output_2'] == result['output'] and annotation['preference'] == 1.0:
                result['alpaca_score'] = 0
                break

# compute correlation
alpaca_eval_scores = [result['alpaca_score'] for result in reward_results if 'alpaca_score' in result]
reward_scores = [result['score'] for result in reward_results if 'alpaca_score' in result]
correlation = np.corrcoef(alpaca_eval_scores, reward_scores)
print(correlation)

# make a boxplot: one for 0.0, one for 1.0
import matplotlib.pyplot as plt

# Separate reward scores by alpaca_score
reward_scores_0 = [result['score'] for result in reward_results if 'alpaca_score' in result and result['alpaca_score'] == 0]
reward_scores_1 = [result['score'] for result in reward_results if 'alpaca_score' in result and result['alpaca_score'] == 1]

# Create boxplot
plt.figure(figsize=(8, 6))
plt.boxplot([reward_scores_0, reward_scores_1], labels=['Alpaca Score 0', 'Alpaca Score 1'])
plt.title('Reward Scores by Alpaca Eval Scores')
plt.ylabel('Reward Scores')
plt.xlabel('Alpaca Eval Scores')
plt.savefig('alpaca_eval_reward_correlation.png')
