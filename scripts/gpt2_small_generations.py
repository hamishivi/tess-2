from transformers import pipeline

generator = pipeline("text-generation", model="gpt2")

# load data
with open("scripts/test_prefixes.txt", "r") as f:
    prefixes = [line.strip() for line in f.readlines()]

# generate
for prefix in prefixes:
    output = generator(
        prefix,
        max_length=256,
        do_sample=True,
        top_p=0.99,
        num_return_sequences=1,
        return_full_text=False,
    )
    print(prefix + "\nGPT-2 OUTPUT\n" + output[0]["generated_text"] + "\n\n")
