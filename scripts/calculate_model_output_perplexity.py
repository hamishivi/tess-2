import json
import logging
import os
import sys

import torch
from datasets import load_dataset
from transformers import (
    MODEL_FOR_MASKED_LM_MAPPING,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    set_seed,
)

from sdlm.arguments import DiffusionArguments, ModelArguments
from sdlm.metrics.metrics import distinct_n_grams, mauve
from sdlm.metrics.perplexity import conditional_perplexity
from sdlm.models import CDCDRobertaConfig, CDCDRobertaForDiffusionLM
from sdlm.pipelines.simplex_ddpm import SimplexDDPMPipeline
from sdlm.schedulers import TokenWiseSimplexDDPMScheduler

logger = logging.getLogger(__name__)
MODEL_CONFIG_CLASSES = list(MODEL_FOR_MASKED_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def main():
    parser = HfArgumentParser((ModelArguments, DiffusionArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, diffusion_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, diffusion_args = parser.parse_args_into_dataclasses()

    # Set seed before initializing model.
    set_seed(42)
    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    config = CDCDRobertaConfig.from_pretrained(
        model_args.model_name_or_path,
        self_condition=diffusion_args.self_condition,
        self_condition_zeros_after_softmax=diffusion_args.self_condition_zeros_after_softmax,
        deepmind_conditional=diffusion_args.deepmind_conditional,
        classifier_free_simplex_inputs=diffusion_args.classifier_free_simplex_inputs,
        classifier_free_uncond_input=diffusion_args.classifier_free_uncond_input,
        self_condition_mlp_projection=diffusion_args.self_condition_mlp_projection,
        self_condition_mix_before_weights=diffusion_args.self_condition_mix_before_weights,
        self_condition_mix_logits_before_weights=diffusion_args.self_condition_mix_logits_before_weights,
        empty_token_be_mask=diffusion_args.empty_token_be_mask,
        **config_kwargs,
    )
    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name, **tokenizer_kwargs
        )
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path, **tokenizer_kwargs
        )
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    if model_args.model_name_or_path:
        model = CDCDRobertaForDiffusionLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    else:
        raise RuntimeError("You need to load a pretrained model")

    # We resize the xs only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    vocab_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > vocab_size:
        model.resize_token_embeddings(len(tokenizer))

    # for some insane reason some of the model is not correctly loaded using from_pretrained...
    state_dict = torch.load(
        os.path.join(model_args.model_name_or_path, "pytorch_model.bin"),
        map_location="cpu",
    )
    # for some insane reason the word embeddings dont get loaded
    model.roberta.embeddings.word_embeddings.weight = torch.nn.Parameter(
        state_dict["roberta.embeddings.word_embeddings.weight"]
    )
    model.tie_weights()
    # make sure loading is entirely correct.
    assert (
        len(
            [k for k in state_dict if torch.any(state_dict[k] != model.state_dict()[k])]
        )
        == 0
    )

    max_eval_samples = 128
    # load eval outputs
    dataset = load_dataset("c4", "en", split="validation", streaming=True)
    # try to keep only longer examples for prompting
    dataset = dataset.filter(lambda x: len(x["text"].split()) > 256)
    dataset = dataset.shuffle(seed=42).take(max_eval_samples)
    # get gold texts
    gold = [tokenizer.decode(tokenizer(x["text"]).input_ids[:512]) for x in dataset]
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
    # if not os.path.exists(f"{model_args.model_name_or_path}-outputs.json"):
    if True:
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
                        tokenizer.decode(output_tokens[256:], skip_special_tokens=False)
                        .split("</s>")[0]
                        .replace("<s>", "")
                        .strip()
                    )
    else:
        with open(f"{model_args.model_name_or_path}-outputs.json", "r") as f:
            results = json.load(f)
            outputs = results["outputs"]
            prefixes = results["prefixes"]
    combined = [p.strip() + " " + o.strip() for p, o in zip(prefixes, outputs)]
    # setup causal model for metrics
    causal_model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B")
    causal_model = causal_model.cuda()
    causal_tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")
    # tmp save outputs
    results = {
        "outputs": outputs,
        "prefixes": prefixes,
    }
    # with open(f"{model_args.model_name_or_path}-outputs.json", "w") as f:
    #     f.write(json.dumps(results, indent=4))
    # quick clean: add a space after the prefix
    filtered_prefixes = [p for i, p in enumerate(prefixes) if p and outputs[i]]
    filtered_outputs = [o for i, o in enumerate(outputs) if o and prefixes[i]]
    prefixes = filtered_prefixes
    outputs = filtered_outputs
    prefixes = [
        p + " "
        for i, p in enumerate(prefixes)
        if prefixes[i][-1] != " " and outputs[i][0] != " "
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
    }
    # save outputs
    # with open(f"{model_args.model_name_or_path}-outputs.json", "w") as f:
    #     f.write(json.dumps(results, indent=4))


if __name__ == "__main__":
    main()
