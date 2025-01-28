import torch
import torch.distributed as dist
import datetime
from torch.nn.parallel import DistributedDataParallel as DDP
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
import os
import sys

logger = logging.getLogger(__name__)

def print_rank(msg, rank):
    """Debug print with rank information"""
    print(f"[Rank {rank}] {msg}", flush=True)

def setup_distributed():
    """Setup for single-node distributed training with shared memory communication"""
    try:
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        world_size = torch.cuda.device_count()

        if world_size > 1:
            print_rank(f"Initializing process group with world size {world_size}", local_rank)
            torch.cuda.set_device(local_rank)
            # Use simpler initialization for same-node GPUs
            dist.init_process_group(
                backend='nccl',
                init_method='env://',
                rank=local_rank,
                world_size=world_size,
                timeout=datetime.timedelta(minutes=60)
            )
            print_rank("Process group initialized", local_rank)
        else:
            print_rank("Running in single GPU mode", local_rank)

        return local_rank, world_size
    except Exception as e:
        print(f"Error in setup_distributed: {str(e)}", flush=True)
        sys.exit(1)

def preprocess(text):
    text = text.strip()
    text = text.replace(" [title]", ". ")
    text = re.sub("\\[.*?\\]", "", text)
    text = text.replace("  ", " ")
    return text

def setup_pipeline(model, tokenizer, diffusion_args, local_rank):
    print_rank("Setting up pipeline", local_rank)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    # Move model to GPU first
    model = model.to(device)
    print_rank("Model moved to device", local_rank)

    # Create pipeline with base model
    pipeline = SimplexDDPMPipelineForEvaluation(
        model=model,
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

    # Store a reference to the base model and wrap it for DDP
    if torch.cuda.device_count() > 1:
        pipeline.base_model = model  # Keep reference to unwrapped model
        ddp_model = DDP(model, device_ids=[local_rank], output_device=local_rank)
        # Add config property to DDP model
        type(ddp_model).config = property(lambda self: pipeline.base_model.config)
        pipeline.model = ddp_model
        print_rank("Model wrapped in DDP", local_rank)

    print_rank("Pipeline setup complete", local_rank)
    return pipeline

def compute_batch_loss(pipeline, inputs, targets):
    tokenizer = pipeline.tokenizer
    device = next(pipeline.model.parameters()).device

    # Use base_model for configuration if available
    model_for_config = getattr(pipeline, 'base_model', pipeline.model)

    inps, masks = [], []
    for input_text, target_text in zip(inputs, targets):
        full_text = input_text.strip() + " " + target_text.strip()
        full_tokenized = tokenizer(full_text)
        input_tokenized = tokenizer(input_text.strip())
        inp = full_tokenized.input_ids
        inp_len = len(input_tokenized.input_ids) - 1
        mask = [0] * inp_len + [1] * (len(inp) - inp_len)
        inps.append(inp)
        masks.append(mask)
    max_len = 2048
    inps_padded = [x + [tokenizer.pad_token_id] * (max_len - len(x)) for x in inps]
    masks_padded = [x + [1] * (max_len - len(x)) for x in masks]

    batch = {
        "input_ids": torch.tensor(inps_padded).to(device),
        "span_mask": torch.tensor(masks_padded).to(device).to(torch.bool),
    }
    timestep_losses = []

    try:
        for out in pipeline(batch=batch, seq_length=max_len):
            loss = out.loss
            #target_mask = batch["span_mask"] == 1
            #target_ids = batch["input_ids"].clone()
            #target_ids[~target_mask] = -100
            #target_ids[target_ids == tokenizer.pad_token_id] = -100
            #target_ids = target_ids[:, 1:]
            #logits = logits[:, :-1]
            #loss = F.cross_entropy(
            #    logits.reshape(-1, logits.size(-1)),
            #    target_ids.reshape(-1),
            #    ignore_index=-100,
            #    reduction='none',
            #)
            #loss = loss.view(target_ids.size(0), -1).mean(dim=1)
            timestep_losses.append(loss.cpu().detach().numpy())
    except Exception as e:
        print(f"Error in compute_batch_loss: {str(e)}", flush=True)
        raise

    avg_loss = np.mean(timestep_losses, axis=0)
    return avg_loss

def get_rank_split(dataset, local_rank, world_size):
    """Split dataset for current GPU"""
    if world_size > 1:
        per_rank = len(dataset) // world_size
        start_idx = local_rank * per_rank
        end_idx = start_idx + per_rank if local_rank != world_size - 1 else len(dataset)
        return dataset.select(range(start_idx, end_idx))
    return dataset

def eval_hellaswag(pipeline, batch_size=2):
    local_rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1

    ds = load_dataset("Rowan/hellaswag", split='validation')
    ds = get_rank_split(ds, local_rank, world_size)

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
    if local_rank == 0:
        print(f"\nStarting HellaSwag evaluation with {total_cnt} examples", flush=True)
        pbar = tqdm(range(0, total_cnt, batch_size), desc="Evaluating HellaSwag")
    else:
        pbar = range(0, total_cnt, batch_size)

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
        batch_correct = np.sum(preds == batch_labels)
        cor += batch_correct

        if local_rank == 0:
            batch_acc = batch_correct / len(batch_labels)
            running_acc = cor / (i + len(batch_labels))
            print(f"Batch {i//batch_size}: acc={batch_acc:.4f}, running_acc={running_acc:.4f} " +
                  f"correct={int(batch_correct)}/{len(batch_labels)}", flush=True)
            pbar.set_postfix({
                'batch_acc': f'{batch_acc:.4f}',
                'running_acc': f'{running_acc:.4f}'
            })

    # Gather results from all processes
    if world_size > 1:
        cor = torch.tensor(cor).cuda()
        dist.all_reduce(cor, op=dist.ReduceOp.SUM)
        total_cnt = torch.tensor(total_cnt).cuda()
        dist.all_reduce(total_cnt, op=dist.ReduceOp.SUM)
        cor = cor.item()
        total_cnt = total_cnt.item()

    final_acc = cor / total_cnt
    if local_rank == 0:
        print(f"\nHellaSwag Final Results:")
        print(f"Total examples: {total_cnt}")
        print(f"Total correct: {int(cor)}")
        print(f"Accuracy: {final_acc:.4f}")
        print("-" * 40)
    return final_acc

def eval_wino(pipeline, batch_size=2):
    local_rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1

    ds = load_dataset("allenai/winogrande", "winogrande_xl", split='validation')
    ds = get_rank_split(ds, local_rank, world_size)

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
    if local_rank == 0:
        print(f"\nStarting Winogrande evaluation with {total_cnt} examples", flush=True)
        pbar = tqdm(range(0, total_cnt, batch_size), desc="Evaluating Winogrande")
    else:
        pbar = range(0, total_cnt, batch_size)

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
        batch_correct = np.sum(preds == batch_labels)
        cor += batch_correct

        if local_rank == 0:
            batch_acc = batch_correct / len(batch_labels)
            running_acc = cor / (i + len(batch_labels))
            print(f"Batch {i//batch_size}: acc={batch_acc:.4f}, running_acc={running_acc:.4f} " +
                  f"correct={int(batch_correct)}/{len(batch_labels)}", flush=True)
            pbar.set_postfix({
                'batch_acc': f'{batch_acc:.4f}',
                'running_acc': f'{running_acc:.4f}'
            })

    # Gather results from all processes
    if world_size > 1:
        cor = torch.tensor(cor).cuda()
        dist.all_reduce(cor, op=dist.ReduceOp.SUM)
        total_cnt = torch.tensor(total_cnt).cuda()
        dist.all_reduce(total_cnt, op=dist.ReduceOp.SUM)
        cor = cor.item()
        total_cnt = total_cnt.item()

    final_acc = cor / total_cnt
    if local_rank == 0:
        print(f"\nWinogrande Final Results:")
        print(f"Total examples: {total_cnt}")
        print(f"Total correct: {int(cor)}")
        print(f"Accuracy: {final_acc:.4f}")
        print("-" * 40)
    return final_acc

def eval_piqa(pipeline, batch_size=2):
    local_rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1

    ds = load_dataset("ybisk/piqa", split='validation')
    ds = get_rank_split(ds, local_rank, world_size)

    all_queries, all_choices, all_labels = [], [], []
    for doc in ds:
        query = f"Question: {doc['goal']}\nAnswer: "
        choices = [doc["sol1"], doc["sol2"]]
        all_queries.append(query)
        all_choices.append(choices)
        all_labels.append(doc["label"])

    total_cnt = len(all_queries)
    cor = 0
    if local_rank == 0:
        print(f"\nStarting PIQA evaluation with {total_cnt} examples", flush=True)
        pbar = tqdm(range(0, total_cnt, batch_size), desc="Evaluating PIQA")
    else:
        pbar = range(0, total_cnt, batch_size)

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
        batch_correct = np.sum(preds == batch_labels)
        cor += batch_correct

        if local_rank == 0:
            batch_acc = batch_correct / len(batch_labels)
            running_acc = cor / (i + len(batch_labels))
            print(f"Batch {i//batch_size}: acc={batch_acc:.4f}, running_acc={running_acc:.4f} " +
                  f"correct={int(batch_correct)}/{len(batch_labels)}", flush=True)
            pbar.set_postfix({
                'batch_acc': f'{batch_acc:.4f}',
                'running_acc': f'{running_acc:.4f}'
            })

    # Gather results from all processes
    if world_size > 1:
        cor = torch.tensor(cor).cuda()
        dist.all_reduce(cor, op=dist.ReduceOp.SUM)
        total_cnt = torch.tensor(total_cnt).cuda()
        dist.all_reduce(total_cnt, op=dist.ReduceOp.SUM)
        cor = cor.item()
        total_cnt = total_cnt.item()

    final_acc = cor / total_cnt
    if local_rank == 0:
        print(f"\nPIQA Final Results:")
        print(f"Total examples: {total_cnt}")
        print(f"Total correct: {int(cor)}")
        print(f"Accuracy: {final_acc:.4f}")
        print("-" * 40)
    return final_acc

def eval_siqa(pipeline, batch_size=2):
    local_rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1

    ds = load_dataset("allenai/social_i_qa", split='validation')
    ds = get_rank_split(ds, local_rank, world_size)

    all_queries, all_choices, all_labels = [], [], []
    for doc in ds:
        query = f"Question: {doc['context']} {doc['question']}\nAnswer: "
        choices = [doc['answerA'], doc['answerB'], doc['answerC']]
        all_queries.append(query)
        all_choices.append(choices)
        all_labels.append(int(doc["label"]) - 1)

    total_cnt = len(all_queries)
    cor = 0
    if local_rank == 0:
        print(f"\nStarting SIQA evaluation with {total_cnt} examples", flush=True)
        pbar = tqdm(range(0, total_cnt, batch_size), desc="Evaluating SIQA")
    else:
        pbar = range(0, total_cnt, batch_size)

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
        batch_correct = np.sum(preds == batch_labels)
        cor += batch_correct

        if local_rank == 0:
            batch_acc = batch_correct / len(batch_labels)
            running_acc = cor / (i + len(batch_labels))
            print(f"Batch {i//batch_size}: acc={batch_acc:.4f}, running_acc={running_acc:.4f} " +
                  f"correct={int(batch_correct)}/{len(batch_labels)}", flush=True)
            pbar.set_postfix({
                'batch_acc': f'{batch_acc:.4f}',
                'running_acc': f'{running_acc:.4f}'
            })

    # Gather results from all processes
    if world_size > 1:
        cor = torch.tensor(cor).cuda()
        dist.all_reduce(cor, op=dist.ReduceOp.SUM)
        total_cnt = torch.tensor(total_cnt).cuda()
        dist.all_reduce(total_cnt, op=dist.ReduceOp.SUM)
        cor = cor.item()
        total_cnt = total_cnt.item()

    final_acc = cor / total_cnt
    if local_rank == 0:
        print(f"\nSIQA Final Results:")
        print(f"Total examples: {total_cnt}")
        print(f"Total correct: {int(cor)}")
        print(f"Accuracy: {final_acc:.4f}")
        print("-" * 40)
    return final_acc

def main():
    try:
        # Initialize distributed setup
        local_rank, world_size = setup_distributed()
        print_rank("Distributed setup complete", local_rank)

        model_args, data_args, training_args, diffusion_args = get_args()
        print_rank("Arguments loaded", local_rank)

        tokenizer, model = load_model(model_args, data_args, training_args, diffusion_args, logger)
        print_rank("Model loaded", local_rank)

        pipeline = setup_pipeline(model, tokenizer, diffusion_args, local_rank)
        print_rank("Pipeline created", local_rank)

        # Run all evaluations
        if local_rank == 0:
            print("\nStarting evaluations...", flush=True)

        #eval_piqa(pipeline)
        eval_wino(pipeline)
        eval_siqa(pipeline)
        eval_piqa(pipeline)
        eval_hellaswag(pipeline)

        # Clean up
        if world_size > 1:
            print_rank("Cleaning up distributed process group", local_rank)
            dist.destroy_process_group()

    except Exception as e:
        print(f"Error in main: {str(e)}", flush=True)
        if world_size > 1:
            dist.destroy_process_group()
        sys.exit(1)

if __name__ == "__main__":
    main()
