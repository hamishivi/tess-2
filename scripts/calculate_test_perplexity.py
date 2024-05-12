import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from sdlm.metrics.perplexity import perplexity

model = AutoModelForCausalLM.from_pretrained("gpt2-xl").cuda()
tokenizer = AutoTokenizer.from_pretrained("gpt2-xl")

c4_val = load_dataset("c4", "en", split="validation", streaming=True)
dataset = c4_val.map(
    lambda examples: tokenizer(examples["text"], max_length=256, truncation=True),
    batched=True,
)

generations = []
prompts = []
for i, sample in tqdm(enumerate(dataset)):
    generation = tokenizer.decode(
        model.generate(
            torch.tensor(sample["input_ids"]).cuda()[None,],
            do_sample=False,
            max_length=256,
        )[0]
    )
    generations.append(generation)
    prompts.append(sample["text"])
    if i >= 512:
        break

print("masked perplexities:")
results = perplexity(generations, model, tokenizer)
print(results)
print("marked perplexities:")
results = perplexity(
    [p + " " + g for p, g in zip(prompts, generations)], model, tokenizer
)
print(results)
