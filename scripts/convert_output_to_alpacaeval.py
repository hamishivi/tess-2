'''
Convert the generated output formats we get to something AlpacaEval can natively understand.
'''
import json
import argparse

parser = argparse.ArgumentParser(description='Convert the generated output formats we get to something AlpacaEval can natively understand.')
parser.add_argument('input', type=str, help='The input file to convert.')
parser.add_argument('output', type=str, help='The output file to write to.')
args = parser.parse_args()

with open(args.input, 'r') as f:
    data = json.load(f)

outputs = data["inference_100_pred_texts_from_logits_masked"]
instructions = data["inference_100_pred_texts_from_logits_marked"]
instructions = [x.split("<|assistant|>\n")[0].replace("<|user|>", "").strip() for x in instructions]

aeval_data = [{"instruction": instructions[i], "output": outputs[i]} for i in range(len(outputs))]
with open(args.output, 'w') as f:
    json.dump(aeval_data, f, indent=4)
