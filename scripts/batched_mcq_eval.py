import torch
from sdlm.pipelines.simplex_ddpm import SimplexDDPMPipelineForEvaluation
from sdlm.schedulers import TokenWiseSimplexDDPMScheduler
import numpy as np
import torch.nn.functional as F
from datasets import load_dataset
from sdlm.arguments import get_args
from sdlm.models.utils import load_model
import logging
import re
from tqdm import tqdm

logger = logging.getLogger(__name__)

def preprocess(text):
    text = text.strip()
    text = text.replace(" [title]", ". ")
    text = re.sub("\\[.*?\\]", "", text)
    text = text.replace("  ", " ")
    return text

def setup_pipeline(model, tokenizer, diffusion_args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline = SimplexDDPMPipelineForEvaluation(
        model=model.to(device),
        scheduler=TokenWiseSimplexDDPMScheduler(
            num_train_timesteps=diffusion_args.num_train_timesteps
            if hasattr(diffusion_args, "num_train_timesteps") else 32,
            beta_schedule=getattr(diffusion_args, "beta_schedule", "squaredcos_improved_ddpm"),
            simplex_value=getattr(diffusion_args, "simplex_value", 5.0),
            clip_sample=getattr(diffusion_args, "clip_sample", False),
            device=device,
        ),
        simplex_value=getattr(diffusion_args, "simplex_value", 5.0),
        top_p=getattr(diffusion_args, "top_p", 0.99),
        sampling_type="top_p",
        is_conditional_generation=True,
        tokenizer=tokenizer,
        classifier_free_uncond_input="empty_token",
        temperature=getattr(diffusion_args, "temperature", 1.0),
        guidance_softmax_combination=True,
    )
    return pipeline

def compute_batch_loss(pipeline, inputs, targets):
    tokenizer = pipeline.tokenizer
    device = next(pipeline.model.parameters()).device
    
    inps, masks = [], []
    for input_text, target_text in zip(inputs, targets):
        full_text = input_text + " " + target_text
        full_tokenized = tokenizer(full_text)
        input_tokenized = tokenizer(input_text)
        inp = full_tokenized.input_ids
        inp_len = len(input_tokenized.input_ids)
        mask = [0] * inp_len + [1] * (len(inp) - inp_len)
        inps.append(inp)
        masks.append(mask)

    max_len = max(len(x) for x in inps)
    inps_padded = [x + [tokenizer.pad_token_id] * (max_len - len(x)) for x in inps]
    masks_padded = [x + [0] * (max_len - len(x)) for x in masks]
    
    batch = {
        "input_ids": torch.tensor(inps_padded).to(device),
        "span_mask": torch.tensor(masks_padded).to(device).to(torch.bool),
    }
    timestep_losses = []
    for out in pipeline(batch=batch, seq_length=max_len):
        logits = out.logits
        target_mask = batch["span_mask"] == 1
        target_ids = batch["input_ids"].clone()
        target_ids[~target_mask] = -100
        target_ids = target_ids[:, 1:]
        logits = logits[:, :-1]
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            target_ids.reshape(-1),
            ignore_index=-100,
            reduction='none',
        )
        loss = loss.view(target_ids.size(0), -1).mean(dim=1)
        timestep_losses.append(loss.cpu().detach().numpy())
    
    avg_loss = np.mean(timestep_losses, axis=0)
    return avg_loss

def eval_hellaswag(pipeline, batch_size=32):
    ds = load_dataset("Rowan/hellaswag", split='validation')
    all_queries, all_choices, all_labels = [], [], []

    for doc in ds:
        ctx = doc["ctx_a"] + " " + doc["ctx_b"].capitalize()
        query = preprocess(doc["activity_label"] + ": " + ctx)
        choices = [preprocess(ending) for ending in doc["endings"]]
        all_queries.append(query)
        all_choices.append(choices)
        all_labels.append(int(doc["label"]))

    total_cnt = len(all_queries)
    cor = 0
    pbar = tqdm(range(0, total_cnt, batch_size), desc="Evaluating HellaSwag")

    for i in pbar:
        batch_queries = all_queries[i:i+batch_size]
        batch_choices = all_choices[i:i+batch_size]
        batch_labels = all_labels[i:i+batch_size]

        input_texts, target_texts = [], []
        for q, chs in zip(batch_queries, batch_choices):
            input_texts.extend([q] * len(chs))
            target_texts.extend(chs)

        losses = compute_batch_loss(pipeline, input_texts, target_texts)
        losses = losses.reshape(-1, 4)
        preds = np.argmin(losses, axis=1)
        cor += np.sum(preds == batch_labels)
        pbar.set_postfix({'acc': f'{cor / (i + batch_size):.4f}'})

    final_acc = cor / total_cnt
    print(f'HellaSwag acc: {final_acc:.4f}')
    return final_acc

def eval_wino(pipeline, batch_size=32):
    ds = load_dataset("allenai/winogrande", "winogrande_xl", split='validation')
    all_prefixes, all_suffixes, all_options, all_labels = [], [], [], []
    answer_to_num = {"1": 0, "2": 1}

    for doc in ds:
        idx = doc["sentence"].index("_")
        prefix = doc["sentence"][:idx]
        suffix = doc["sentence"][idx+1:].strip()
        options = [doc["option1"], doc["option2"]]
        all_prefixes.append(prefix)
        all_suffixes.append(suffix)
        all_options.append(options)
        all_labels.append(answer_to_num[doc["answer"]])

    total_cnt = len(all_prefixes)
    cor = 0
    pbar = tqdm(range(0, total_cnt, batch_size), desc="Evaluating Winogrande")

    for i in pbar:
        batch_prefixes = all_prefixes[i:i+batch_size]
        batch_suffixes = all_suffixes[i:i+batch_size]
        batch_options = all_options[i:i+batch_size]
        batch_labels = all_labels[i:i+batch_size]

        input_texts, target_texts = [], []
        for p, s, opts in zip(batch_prefixes, batch_suffixes, batch_options):
            targets = [opt + s for opt in opts]
            input_texts.extend([p] * len(opts))
            target_texts.extend(targets)

        losses = compute_batch_loss(pipeline, input_texts, target_texts)
        losses = losses.reshape(-1, 2)
        preds = np.argmin(losses, axis=1)
        cor += np.sum(preds == batch_labels)
        pbar.set_postfix({'acc': f'{cor / (i + batch_size):.4f}'})

    final_acc = cor / total_cnt
    print(f'Winogrande acc: {final_acc:.4f}')
    return final_acc

def eval_piqa(pipeline, batch_size=32):
    ds = load_dataset("ybisk/piqa", split='validation')
    all_queries, all_choices, all_labels = [], [], []

    for doc in ds:
        query = f"<|user|>\nQuestion: {doc['goal']}\n<|assistant|>\nAnswer: "
        choices = [doc["sol1"], doc["sol2"]]
        all_queries.append(query)
        all_choices.append(choices)
        all_labels.append(doc["label"])

    total_cnt = len(all_queries)
    cor = 0
    pbar = tqdm(range(0, total_cnt, batch_size), desc="Evaluating PIQA")

    for i in pbar:
        batch_queries = all_queries[i:i+batch_size]
        batch_choices = all_choices[i:i+batch_size]
        batch_labels = all_labels[i:i+batch_size]

        input_texts, target_texts = [], []
        for q, chs in zip(batch_queries, batch_choices):
            input_texts.extend([q] * len(chs))
            target_texts.extend(chs)

        losses = compute_batch_loss(pipeline, input_texts, target_texts)
        losses = losses.reshape(-1, 2)
        preds = np.argmin(losses, axis=1)
        cor += np.sum(preds == batch_labels)
        pbar.set_postfix({'acc': f'{cor / (i + batch_size):.4f}'})

    final_acc = cor / total_cnt
    print(f'PIQA acc: {final_acc:.4f}')
    return final_acc

def eval_siqa(pipeline, batch_size=32):
    ds = load_dataset("allenai/social_i_qa", split='validation')
    all_queries, all_choices, all_labels = [], [], []

    for doc in ds:
        query = f"Question: {doc['context']} {doc['question']}\nAnswer: "
        choices = [doc['answerA'], doc['answerB'], doc['answerC']]
        all_queries.append(query)
        all_choices.append(choices)
        all_labels.append(int(doc["label"]) - 1)

    total_cnt = len(all_queries)
    cor = 0
    pbar = tqdm(range(0, total_cnt, batch_size), desc="Evaluating SIQA")

    for i in pbar:
        batch_queries = all_queries[i:i+batch_size]
        batch_choices = all_choices[i:i+batch_size]
        batch_labels = all_labels[i:i+batch_size]

        input_texts, target_texts = [], []
        for q, chs in zip(batch_queries, batch_choices):
            input_texts.extend([q] * len(chs))
            target_texts.extend(chs)

        losses = compute_batch_loss(pipeline, input_texts, target_texts)
        losses = losses.reshape(-1, 3)
        preds = np.argmin(losses, axis=1)
        cor += np.sum(preds == batch_labels)
        pbar.set_postfix({'acc': f'{cor / (i + batch_size):.4f}'})

    final_acc = cor / total_cnt
    print(f'SIQA acc: {final_acc:.4f}')
    return final_acc

def main():
    model_args, data_args, training_args, diffusion_args = get_args()
    tokenizer, model = load_model(model_args, data_args, training_args, diffusion_args, logger)

    pipeline = setup_pipeline(model, tokenizer, diffusion_args)
    #eval_hellaswag(pipeline)
    eval_piqa(pipeline)
    eval_wino(pipeline)
    eval_siqa(pipeline)

if __name__ == "__main__":
    main()
