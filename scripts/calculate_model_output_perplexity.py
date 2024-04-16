import json
import logging
import os

import torch
from datasets import load_dataset
from transformers import (
    MODEL_FOR_MASKED_LM_MAPPING,
    AutoModelForCausalLM,
    AutoTokenizer,
)

from sdlm.arguments import get_args
from sdlm.metrics.metrics import distinct_n_grams, mauve
from sdlm.metrics.perplexity import conditional_perplexity
from sdlm.models.utils import load_model
from sdlm.pipelines.simplex_ddpm import SimplexDDPMPipeline
from sdlm.schedulers import TokenWiseSimplexDDPMScheduler

logger = logging.getLogger(__name__)
MODEL_CONFIG_CLASSES = list(MODEL_FOR_MASKED_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def main():
    model_args, data_args, training_args, diffusion_args = get_args()
    tokenizer, model = load_model(
        model_args, data_args, training_args, diffusion_args, logger
    )

    # for some insane reason some of the model is not correctly loaded using from_pretrained...
    # state_dict = torch.load(
    #     os.path.join(model_args.model_name_or_path, "pytorch_model.bin"),
    #     map_location="cpu",
    # )
    # # for some insane reason the word embeddings dont get loaded
    # model.roberta.embeddings.word_embeddings.weight = torch.nn.Parameter(
    #     state_dict["roberta.embeddings.word_embeddings.weight"]
    # )
    # model.tie_weights()
    # make sure loading is entirely correct.
    # assert (
    #     len(
    #         [k for k in state_dict if torch.any(state_dict[k] != model.state_dict()[k])]
    #     )
    #     == 0
    # )

    max_eval_samples = 512
    # load eval outputs
    dataset = load_dataset("c4", "en", split="validation", streaming=True)
    # try to keep only longer examples for prompting
    dataset = dataset.filter(lambda x: len(x["text"].split()) > 256)
    dataset = dataset.shuffle(seed=42).take(max_eval_samples)
    # get gold texts
    gold = [tokenizer.decode(tokenizer(x["text"]).input_ids[:512]) for x in dataset]
    gold_output_only = [
        tokenizer.decode(tokenizer(x["text"]).input_ids[256:512]) for x in dataset
    ]
    # some constants for generations
    simplex_value = 5.0
    top_p = 1.0
    temperature = 1.0
    diffusion_steps = 100
    beta_schedule = "squaredcos_improved_ddpm"
    clip_sample = False
    guidance_scale = 1.0
    generated_sequence_length = 256

    # tokenize and setup pipeline
    def tokenize_and_pad(examples):
        inputs = tokenizer(
            examples["text"],
            truncation=True,
            max_length=512,
            padding="max_length",
            return_tensors="pt",
        )
        # we will generate from the first 256 tokens.
        inputs = torch.cat(
            [
                inputs["input_ids"][:, :256],
                torch.ones((inputs["input_ids"].shape[0], 256), dtype=torch.long)
                * tokenizer.pad_token_id,
            ],
            dim=1,
        )
        span_mask = inputs == tokenizer.pad_token_id
        return {"input_ids": inputs, "span_mask": span_mask}

    dataset = dataset.map(tokenize_and_pad, batched=False)

    model.eval()
    # setup pipeline for generation
    pipeline = SimplexDDPMPipeline(
        model=model.cuda(),
        scheduler=TokenWiseSimplexDDPMScheduler(
            num_train_timesteps=diffusion_steps,
            beta_schedule=beta_schedule,
            simplex_value=simplex_value,
            clip_sample=clip_sample,
            device=torch.device("cuda", 0),
        ),
        simplex_value=simplex_value,
        top_p=top_p,
        sampling_type="top_p",  # currently only this is supported
        is_conditional_generation=True,
        tokenizer=tokenizer,
        classifier_free_uncond_input="empty_token",
        temperature=temperature,
        guidance_softmax_combination=True,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=24,
        shuffle=False,
    )
    outputs = []
    prefixes = []
    if not os.path.exists(f"{model_args.model_name_or_path}-outputs.json"):
        # AR model path
        if False:
            from transformers import pipeline

            ar_generator = pipeline(model="gpt2", device=0)
            with torch.inference_mode():
                for batch in dataloader:
                    for input_tokens in batch["input_ids"]:
                        prefixes.append(
                            tokenizer.decode(
                                input_tokens.squeeze()[:256], skip_special_tokens=True
                            )
                        )
                        output = ar_generator(
                            tokenizer.decode(
                                input_tokens.squeeze()[:256], skip_special_tokens=True
                            ),
                            max_length=generated_sequence_length + 256,
                            do_sample=True,
                            top_p=0.99,
                            return_full_text=False,
                        )
                        outputs.append(output[0]["generated_text"])
        elif True:
            with torch.inference_mode():
                for batch in dataloader:
                    for input_tokens in batch["input_ids"]:
                        prefixes.append(
                            tokenizer.decode(
                                input_tokens.squeeze()[:256], skip_special_tokens=True
                            )
                        )
                    # yield over until end.
                    for o in pipeline(
                        batch={
                            "input_ids": batch["input_ids"].squeeze().cuda(),
                            "span_mask": batch["span_mask"].squeeze().cuda(),
                        },
                        guidance_scale=guidance_scale,
                        seq_length=generated_sequence_length,
                    ):
                        output = o
                    for output_tokens in output.logits.argmax(-1):
                        outputs.append(
                            tokenizer.decode(
                                output_tokens[256:], skip_special_tokens=False
                            )
                            .split("</s>")[0]
                            .replace("<s>", "")
                            .strip()
                        )
    else:
        with open(f"{model_args.model_name_or_path}-outputs-500steps.json", "r") as f:
            results = json.load(f)
            outputs = results["outputs"]
            prefixes = results["prefixes"]
    combined = [p.strip() + " " + o.strip() for p, o in zip(prefixes, outputs)]
    # gold_combined = [
    #     p.strip() + " " + o.strip() for p, o in zip(prefixes, gold_output_only)
    # ]
    # setup causal model for metrics
    causal_model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B")
    causal_model = causal_model.cuda()
    causal_tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")
    # tmp save outputs
    results = {
        "outputs": outputs,
        "prefixes": prefixes,
    }
    with open("outputs.json", "w") as f:
        f.write(json.dumps(results, indent=4))
    # quick clean: add a space after the prefix
    filtered_prefixes = [p for i, p in enumerate(prefixes) if p and outputs[i]]
    filtered_outputs = [o for i, o in enumerate(outputs) if o and prefixes[i]]
    prefixes = filtered_prefixes
    outputs = filtered_outputs
    prefixes = [
        p + " " if prefixes[i][-1] != " " and gold_output_only[i][0] != " " else p
        for i, p in enumerate(prefixes)
    ]
    # okay! metrics time!
    perplexity_scores = conditional_perplexity(
        outputs, prefixes, causal_model, causal_tokenizer
    )
    ngram_scores = distinct_n_grams(outputs)
    mauve_scores = mauve(
        predictions=combined, references=gold, length=generated_sequence_length
    )

    # taken from CDCD / strudel et al paper
    def unigram_entropy(outputs):
        tokenized_outputs = [tokenizer.encode(x, return_tensors="pt") for x in outputs]
        entropies = []
        for output in tokenized_outputs:
            _, counts = torch.unique(output, return_counts=True, dim=1)
            total_counts = counts.sum()
            probs = counts / total_counts
            entropy = -(probs * torch.log2(probs)).sum()
            entropies.append(entropy)
        return torch.stack(entropies).mean().item()

    entropy_scores = unigram_entropy(outputs)
    print("Total samples: ", len(outputs))
    print("Perplexity: ", perplexity_scores["mean_perplexity"])
    print("dist-1: ", ngram_scores["dist-1"])
    print("dist-2: ", ngram_scores["dist-2"])
    print("dist-3: ", ngram_scores["dist-3"])
    print("dist-4: ", ngram_scores["dist-4"])
    print("Mauve: ", mauve_scores["mauve"])
    print("Entropy: ", entropy_scores)

    results = {
        "perplexity": perplexity_scores["mean_perplexity"],
        "dist-1": ngram_scores["dist-1"],
        "dist-2": ngram_scores["dist-2"],
        "dist-3": ngram_scores["dist-3"],
        "dist-4": ngram_scores["dist-4"],
        "mauve": mauve_scores["mauve"],
        "entropy": entropy_scores,
        "outputs": outputs,
        "prefixes": prefixes,
        "full_outputs": [x + "***" + y + "***" for x, y in zip(prefixes, outputs)],
    }
    # save outputs
    with open("outputs.json", "w") as f:
        f.write(json.dumps(results, indent=4))


if __name__ == "__main__":
    main()
