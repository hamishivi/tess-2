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
    # NOTE: Brackets are artifacts of the WikiHow dataset portion of HellaSwag.
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
            if hasattr(diffusion_args, "num_train_timesteps") else 10,
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
    
    batch_size = len(inputs)
    losses = []
    
    inps, masks = [], []
    for input_text, target_text in zip(inputs, targets):
        inp = tokenizer(input_text + " "  + target_text).input_ids
        inp_len = len(tokenizer(input_text).input_ids)
        mask = [0] * inp_len + [1] * (len(inp) - inp_len)
        inps.append(inp)
        masks.append(mask)

    # stack with padding
    max_len = max([len(x) for x in inps])
    inps = [x + [tokenizer.pad_token_id] * (max_len - len(x)) for x in inps]
    masks = [x + [0] * (max_len - len(x)) for x in masks]
    
    batch = {
        "input_ids": torch.tensor(inps).to(device),
        "span_mask": torch.tensor(masks).to(device).to(torch.bool),
    }
    
    timestep_losses = []
    for out in pipeline(batch=batch, seq_length=max_len):
        logits = out.logits
        target_mask = batch["span_mask"] == 1
        target_ids = batch["input_ids"].clone()
        target_ids[~target_mask] = -100
        bsz = target_ids.size(0)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            target_ids.view(-1),
            ignore_index=-100,
            reduction='none',
        )
        loss = loss.view(bsz, -1).mean(dim=-1)
        timestep_losses.append(loss.cpu().detach().numpy())

    # average over timesteps
    losses.append(np.mean(timestep_losses, axis=1))
    
    return losses

def eval_hellaswag(pipeline):
    ds = load_dataset("Rowan/hellaswag", split='validation')
    total_cnt = cor = 0
    
    for doc in tqdm(ds):
        total_cnt += 1
        ctx = doc["ctx_a"] + " " + doc["ctx_b"].capitalize()
        query = preprocess(doc["activity_label"] + ": " + ctx)
        choices = [preprocess(ending) for ending in doc["endings"]]
        
        score_list = compute_batch_loss(pipeline, [query] * len(choices), choices)
        if np.argmin(score_list) == int(doc["label"]):
            cor += 1
            
    print('HellaSwag acc:', cor/total_cnt)

def eval_wino(pipeline):
    ds = load_dataset("allenai/winogrande", "winogrande_xl", split='validation')
    total_cnt = cor = 0
    
    for doc in tqdm(ds):
        total_cnt += 1
        idx = doc["sentence"].index("_")
        options = [doc["option1"], doc["option2"]]
        answer_to_num = {"1": 0, "2": 1}
        gold = answer_to_num[doc["answer"]]
        
        score_list = []
        for opt in options:
            prefix = doc["sentence"][:idx]
            suffix = doc["sentence"][idx+1:].strip()
            prefix = f"<|user|>\n{prefix}"
            suffix = f"{suffix}\n<|assistant|>\nAnswer: "
            score_list = compute_batch_loss(pipeline, [prefix] * len(options), [opt + suffix for opt in options])
            
        if np.argmin(score_list) == gold:
            cor += 1
            
    print('Winogrande acc:', cor/total_cnt)

def eval_piqa(pipeline):
    ds = load_dataset("ybisk/piqa", split='validation')
    total_cnt = cor = 0
    
    for doc in tqdm(ds):
        total_cnt += 1
        query = f"<|user|>\nQuestion: {doc['goal']}\n<|assistant|>\nAnswer: "
        choices = [doc["sol1"], doc["sol2"]]
        
        score_list = compute_batch_loss(pipeline, [query] * len(choices), choices)
        if np.argmin(score_list) == doc["label"]:
            cor += 1
            
    print('PIQA acc:', cor/total_cnt)

def eval_siqa(pipeline):
    ds = load_dataset("allenai/social_i_qa", split='validation')
    total_cnt = cor = 0

    for doc in tqdm(ds):
        total_cnt += 1
        query = f"<|user|>\nQuestion: {doc['context']} {doc['question']}\n<|assistant|>\nAnswer: "
        choices = [doc['answerA'], doc['answerB'], doc['answerC']]
        gold = int(doc["label"]) - 1

        score_list = compute_batch_loss(pipeline, [query] * len(choices), choices)
        if np.argmin(score_list) == gold:
            cor += 1

    print('SIQA acc:', cor/total_cnt)

def main():
    model_args, data_args, training_args, diffusion_args = get_args()
    tokenizer, model = load_model(model_args, data_args, training_args, diffusion_args, logger)

    pipeline = setup_pipeline(model, tokenizer, diffusion_args)
    eval_hellaswag(pipeline)
    eval_piqa(pipeline)
    eval_wino(pipeline)
    eval_siqa(pipeline)

if __name__ == "__main__":
    main()