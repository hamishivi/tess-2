# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
python examples/scripts/reward_modeling.py \
    --model_name_or_path=facebook/opt-350m \
    --output_dir="reward_modeling_anthropic_hh" \
    --per_device_train_batch_size=16 \
    --num_train_epochs=1 \
    --gradient_accumulation_steps=2 \
    --gradient_checkpointing=True \
    --learning_rate=1.41e-5 \
    --report_to="wandb" \
    --remove_unused_columns=False \
    --optim="adamw_torch" \
    --logging_steps=10 \
    --eval_strategy="steps" \
    --eval_steps=500 \
    --max_length=512 \
"""
import warnings
from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, HfArgumentParser, PreTrainedModel
from transformers.trainer_pt_utils import nested_detach

from trl import ModelConfig, RewardConfig, RewardTrainer, get_kbit_device_map, get_peft_config, get_quantization_config
from sdlm.models.mistral.modeling_mistral import MistralforSequenceClassificationWithPadding
from sdlm.models.utils import get_torch_dtype
from sdlm.schedulers import SimplexDDPMScheduler

tqdm.pandas()

# TODO: allow end_lr to be changed via some config.
def _get_linear_schedule_with_warmup_lr_lambda(current_step: int, *, num_warmup_steps: int, num_training_steps: int, end_lr_ratio: float = 0.1):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    else:
        return end_lr_ratio + (1.0 - end_lr_ratio) * max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))

def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, end_lr_ratio, last_epoch=-1):
    lr_lambda = partial(
        _get_linear_schedule_with_warmup_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        end_lr_ratio=end_lr_ratio,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)


# new little trainer with the scheduler we want.
class RewardTrainerScheduler(RewardTrainer):
    def __init__(self, *args, train_on_noisy_inputs=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_on_noisy_inputs = train_on_noisy_inputs
        self.noise_scheduler = SimplexDDPMScheduler(
        num_train_timesteps=5000,
        beta_schedule="squaredcos_improved_ddpm",
        simplex_value=5,
        clip_sample=False,
        device='cuda',
    )


    def create_scheduler(self, num_training_steps: int, optimizer: torch.optim.Optimizer = None):
        if self.lr_scheduler is None:
            self.lr_scheduler = get_linear_schedule_with_warmup(optimizer, self.args.warmup_steps, num_training_steps, end_lr_ratio=0.1)
            self._created_lr_scheduler = True
        return self.lr_scheduler


    def prediction_step(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        with torch.no_grad():
            loss, logits_dict = self.compute_loss(model, inputs, return_outputs=True)

        if prediction_loss_only:
            return (loss, None, None)

        loss = loss.detach()
        logits = tuple(v for k, v in logits_dict.items() if k not in ignore_keys)
        logits = nested_detach(logits)
        # Stack accepted against rejected, mean over logits
        # and softmax to get preferences between accepted and rejected to sum to 1
        # logits = torch.stack(logits).mean(dim=2).softmax(dim=0).T
        # removing softmax for now, since I want to see the raw logits.
        logits = torch.stack(logits).mean(dim=2).T

        labels = torch.zeros(logits.shape[0])
        labels = self._prepare_inputs(labels)

        return loss, logits, labels


    # hacky override to set cache to false
    # required to fix FA2 + mistral issues
    # see https://github.com/huggingface/trl/issues/1217
    def compute_loss(
            self,
            model,
            inputs,
            return_outputs=False,
        ):
            if not self.use_reward_data_collator:
                warnings.warn(
                    "The current compute_loss is implemented for RewardDataCollatorWithPadding,"
                    " if you are using a custom data collator make sure you know what you are doing or"
                    " implement your own compute_loss method."
                )
            if self.train_on_noisy_inputs:
                from sdlm.utils import convert_to_simplex
                def construct_noisy_simplex(input_ids):
                    # hardcoded simplex value for now TODO: make this a config
                    simplex = convert_to_simplex(
                        input_ids, 5, len(self.tokenizer)
                    )
                    noise = 5 * torch.randn(
                        simplex.shape, device=simplex.device, dtype=torch.float32
                    )
                    bsz = simplex.shape[0]
                    timesteps = torch.randint(
                        0,
                        5000,  # hardcoded value for now TODO: make this a config
                        (bsz, input_ids.shape[1])
                        if False  # is_tokenwise_cdcd_check(self.model)
                        else (bsz,),
                        device=simplex.device,
                        dtype=torch.int64,
                    )
                    timesteps = timesteps[:, None].expand(-1, input_ids.shape[1])
                    # Adds noise to each simplex representation (Forward diffusion process).
                    noisy_simplex = self.noise_scheduler.add_noise(simplex, noise, timesteps)
                    return noisy_simplex
                
                simplex_chosen = construct_noisy_simplex(inputs["input_ids_chosen"])
                simplex_chosen = torch.softmax(simplex_chosen, dim=-1).to(torch.bfloat16)
                inputs_embeds_chosen = F.linear(
                    simplex_chosen, model.get_input_embeddings().weight.data.T
                )
                simplex_rejected = construct_noisy_simplex(inputs["input_ids_rejected"])
                simplex_rejected = torch.softmax(simplex_rejected, dim=-1).to(torch.bfloat16)
                inputs_embeds_rejected = F.linear(
                    simplex_rejected, model.get_input_embeddings().weight.data.T
                )
                rewards_chosen = model(
                    inputs_embeds=inputs_embeds_chosen,
                    attention_mask=inputs["attention_mask_chosen"],
                    return_dict=True,
                    use_cache=False,
                )["logits"]
                rewards_rejected = model(
                    inputs_embeds=inputs_embeds_rejected,
                    attention_mask=inputs["attention_mask_rejected"],
                    return_dict=True,
                    use_cache=False,
                )["logits"]
            else:
                rewards_chosen = model(
                    inputs_ids=inputs["input_ids_chosen"],
                    attention_mask=inputs["attention_mask_chosen"],
                    return_dict=True,
                    use_cache=False,
                )["logits"]
                rewards_rejected = model(
                    inputs_ids=inputs["input_ids_rejected"],
                    attention_mask=inputs["attention_mask_rejected"],
                    return_dict=True,
                    use_cache=False,
                )["logits"]
            # calculate loss, optionally modulate with margin
            if "margin" in inputs:
                loss = -nn.functional.logsigmoid(rewards_chosen - rewards_rejected - inputs["margin"]).mean()
            else:
                loss = -nn.functional.logsigmoid(rewards_chosen - rewards_rejected).mean()

            if return_outputs:
                return loss, {
                    "rewards_chosen": rewards_chosen,
                    "rewards_rejected": rewards_rejected,
                }
            return loss

@dataclass
class RewardModelingArguments:
    include_padding: bool = False  # if true, we pad the input_ids to the max_length and compute reward at final token.
    use_tulu_chat_template: bool = False  # if true, we use the tulu chat template for the input_ids.
    end_lr: float = 1e-6  # final learning rate for the learning rate scheduler.
    dataset_name: str = "argilla/ultrafeedback-binarized-preferences-cleaned"  # dataset to use for reward modeling.
    use_flash_attention2: bool = False  # if true, we use the flash attention2 implementation.
    eval_only: bool = False  # if true, we only evaluate the model.
    train_on_noisy_inputs: bool = False  # if true, we emulate the diffusion noise as input during training.

if __name__ == "__main__":
    parser = HfArgumentParser((RewardConfig, ModelConfig, RewardModelingArguments))
    config, model_config, reward_config = parser.parse_args_into_dataclasses()
    config.gradient_checkpointing_kwargs = dict(use_reentrant=False)

    ################
    # Model & Tokenizer
    ################
    torch_dtype = (
        model_config.torch_dtype
        if model_config.torch_dtype in ["auto", None]
        else getattr(torch, model_config.torch_dtype)
    )
    quantization_config = get_quantization_config(model_config)
    model_kwargs = dict(
        revision=model_config.model_revision,
        trust_remote_code=model_config.trust_remote_code,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
        attn_implementation="flash_attention_2"
            if reward_config.use_flash_attention2
            else "eager",
        torch_dtype=get_torch_dtype(config),
    )
    tokenizer = AutoTokenizer.from_pretrained(model_config.model_name_or_path, revision=model_config.model_revision, use_fast=True)
    # just always add the pad token.
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    # make sure the pad token is set correctly.
    tokenizer.pad_token = "[PAD]"
    tokenizer.pad_token_id = 32000

    if reward_config.use_tulu_chat_template:
        tokenizer.chat_template = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"
    if reward_config.include_padding:
        model = MistralforSequenceClassificationWithPadding.from_pretrained(
            model_config.model_name_or_path, num_labels=1, **model_kwargs
        )
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_config.model_name_or_path, num_labels=1, **model_kwargs
        )

    # resize model embeddings
    vocab_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > vocab_size:
        model.resize_token_embeddings(len(tokenizer))

    # make sure the model knows the pad token id
    model.config.pad_token_id = tokenizer.pad_token_id
    tokenizer.padding_side = "right"

    if model_config.lora_task_type != "SEQ_CLS":
        warnings.warn(
            "You are using a `task_type` that is different than `SEQ_CLS` for PEFT. This will lead to silent bugs"
            " Make sure to pass --lora_task_type SEQ_CLS when using this script."
        )

    # Dataset loading
    raw_datasets = load_dataset(reward_config.dataset_name)
    # use reward bench for validation.
    eval_dataset = load_dataset("allenai/reward-bench", split="filtered")
    # Tokenize chosen/rejected pairs of inputs
    # Adapt this section to your needs for custom datasets

    def preprocess_function(examples):
        new_examples = {
            "input_ids_chosen": [],
            "attention_mask_chosen": [],
            "input_ids_rejected": [],
            "attention_mask_rejected": [],
        }
        for chosen, rejected in zip(examples["chosen"], examples["rejected"]):
            # flatten from 2d to 1d
            tokenize_func = lambda x: tokenizer.apply_chat_template(
                x,
                return_tensors="pt",
                max_length=config.max_length,
                padding=reward_config.include_padding,
            ).flatten()
            tokenized_chosen = tokenize_func(chosen)
            tokenized_rejected = tokenize_func(rejected)
            new_examples["input_ids_chosen"].append(tokenized_chosen)
            new_examples["attention_mask_chosen"].append(torch.ones_like(tokenized_chosen))
            new_examples["input_ids_rejected"].append(tokenized_rejected)
            new_examples["attention_mask_rejected"].append(torch.ones_like(tokenized_rejected))
        return new_examples
    
    def preprocess_function_no_list(examples):
        new_examples = {
            "input_ids_chosen": [],
            "attention_mask_chosen": [],
            "input_ids_rejected": [],
            "attention_mask_rejected": [],
        }
        for prompt, chosen, rejected in zip(examples["prompt"], examples["chosen"], examples["rejected"]):
            # construct lists
            chosen = [{"role": "user", "content": prompt}, {"role": "assistant", "content": chosen}]
            rejected = [{"role": "user", "content": prompt}, {"role": "assistant", "content": rejected}]
            # same as above
            tokenize_func = lambda x: tokenizer.apply_chat_template(
                x,
                return_tensors="pt",
                max_length=config.max_length,
                padding=reward_config.include_padding,
            ).flatten()
            tokenized_chosen = tokenize_func(chosen)
            tokenized_rejected = tokenize_func(rejected)
            new_examples["input_ids_chosen"].append(tokenized_chosen)
            new_examples["attention_mask_chosen"].append(torch.ones_like(tokenized_chosen))
            new_examples["input_ids_rejected"].append(tokenized_rejected)
            new_examples["attention_mask_rejected"].append(torch.ones_like(tokenized_rejected))

        return new_examples

    # Preprocess the dataset and filter out examples that are longer than args.max_length
    raw_datasets = raw_datasets.map(
        preprocess_function,
        batched=True,
        num_proc=4,
    )
    raw_datasets = raw_datasets.filter(
        lambda x: len(x["input_ids_chosen"]) <= config.max_length and len(x["input_ids_rejected"]) <= config.max_length
    )
    train_dataset = raw_datasets["train"]
    eval_dataset = eval_dataset.map(preprocess_function_no_list, batched=True, num_proc=4)
    eval_dataset = eval_dataset.filter(
        lambda x: len(x["input_ids_chosen"]) <= config.max_length and len(x["input_ids_rejected"]) <= config.max_length
    )

    ################
    # Training
    ################
    trainer = RewardTrainerScheduler(
        model=model,
        tokenizer=tokenizer,
        args=config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=get_peft_config(model_config),
        train_on_noisy_inputs=reward_config.train_on_noisy_inputs,
    )
    if not reward_config.eval_only:
        trainer.train()
        trainer.save_model(config.output_dir)
    metrics = trainer.evaluate()
    trainer.log_metrics("eval", metrics)
    print(metrics)
