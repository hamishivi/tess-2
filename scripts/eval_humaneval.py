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
            if hasattr(diffusion_args, "num_train_timesteps") else 100,
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


def eval_humaneval(pipeline):
    from human_eval_infilling.data import write_jsonl, read_problems
    subtasks = "single-line"
    problems = read_problems(benchmark_name=subtasks)
    samples = []
    tokenizer = pipeline.tokenizer
    for task_id in problems:
        prompt = problems[task_id]["prompt"]
        suffix = problems[task_id]["suffix"]
        middle = problems[task_id]["canonical_solution"]

        prefix = tokenizer.encode(prompt, add_special_tokens=False)
        suff = tokenizer.encode(suffix, add_special_tokens=False)
        x0 = prefix + tokenizer.encode(middle, add_special_tokens=False) + suff
        src_mask = [1]*len(prefix)+[0]*(len(x0)-len(prefix)-len(suff))+[1]*len(suff)
        # from diffullama code.
        if len(x0) > 1000:
            print(task_id)
            continue
        
        inputs = {"input_ids": torch.tensor([x0]).cuda(), "span_mask": torch.tensor([src_mask]).bool().cuda()}
        
        out = pipeline(batch=inputs,)
        for x in out:
            res = x.logits.argmax(dim=-1)
        pred = tokenizer.decode(res.tolist()[0][len(prefix)-1:len(x0)-len(suff)-1])
    
        samples.append(dict(task_id=task_id, completion=pred))

    write_jsonl(f"humaneval_{subtasks}.jsonl", samples)
    print("Then run `python -m human_eval_infilling.evaluate humaneval_{subtasks}.jsonl` to evaluate the model.")


def main():
    model_args, data_args, training_args, diffusion_args = get_args()
    tokenizer, model = load_model(model_args, data_args, training_args, diffusion_args, logger)

    pipeline = setup_pipeline(model, tokenizer, diffusion_args)
    eval_humaneval(pipeline)

if __name__ == "__main__":
    main()