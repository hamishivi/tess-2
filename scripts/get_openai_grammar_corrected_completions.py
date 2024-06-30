import argparse
import json
import re

from openai import OpenAI
from tqdm import tqdm
import datasets

parser = argparse.ArgumentParser(description='Get completions for prompts using OpenAI GPT-4')
parser.add_argument('--input_file', type=str, default='prompts.json', help='Path to the file containing prompts')
parser.add_argument('--output_file', type=str, default='completions.json', help='Path to the file to write completions')
parser.add_argument('--model', type=str, default='gpt-4-turbo', help='Model to use for completion')
parser.add_argument('--max_tokens', type=int, default=4096, help='Maximum number of tokens in the completion')
args = parser.parse_args()

GRAMMAR_CORRECTION_PROMPT = "Fix any misspellings or inconsistent formatting errors in the following text. Please output the corrected text and only the text alone. Only edit words when it is clear they are unfinished. fix things like inconsistent use of newlines, or numbered items that do not increment correctly. do NOT fix grammer errors. do NOT remove repetitive language.\n\n"

def get_completions(prompts, model="gpt-4-turbo", max_tokens=4096):
    completions = []
    client = OpenAI()
    for prompt in tqdm(prompts):
        response = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model=model,
            max_tokens=max_tokens,
        )
        completions.append(response.choices[0].message.content.strip())
        print('----- INPUT -------')
        print(prompt)
        print('----- OUTPUT ------')
        print(response.choices[0].message.content)
        print('-------------------')
    return completions

def read_prompts_from_file(file_path):
    with open(file_path, 'r') as file:
        prompts = [line.strip() for line in file.readlines()]
    return prompts

def write_completions_to_file(completions, inputs, file_path):
    # write out in native alpacaeval format
    outputs = []
    for inp, completion in zip(inputs, completions):
        outputs.append({
            "instruction": inp,
            "output": completion,
            "generator": "tess_grammar_correction",
        })
    with open(file_path, 'w') as file:
        json.dump(outputs, file)

def extract_text(sample):
    # Extracting text between \n and \n
    text_between_newlines = re.findall(r'\n(.*?)\n', sample, re.DOTALL)[0]
    # Extracting text after ***
    text_after_stars = re.findall(r'\*\*\*(.*?)\*\*\*', sample, re.DOTALL)
    # Removing *** from the extracted text
    text_after_stars_cleaned = [text.strip() for text in text_after_stars][0]
    return text_between_newlines.strip(), text_after_stars_cleaned.strip()


def main():
    input_file = args.input_file
    output_file = args.output_file

    # assume data is in alpacaeval format
    data = json.load(open(input_file, 'r'))
    outputs = data["inference_250_pred_texts_from_logits_masked"]
    inc_prompts = data["inference_250_pred_texts_from_logits_marked"]
    prompts = [GRAMMAR_CORRECTION_PROMPT + d for d in outputs]
    # call openai and cleanup
    completions = get_completions(prompts)

    inps, outs = list(zip(*[extract_text(sample) for sample in inc_prompts]))

    # write out and done
    write_completions_to_file(completions, inps, output_file)
    print(f"Completions written to {output_file}")

if __name__ == "__main__":
    main()
