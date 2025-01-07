#!/bin/bash

python -m scripts.interactive_demo \
        --model_name_or_path $1 \
        --simplex_value 5 \
        --num_diffusion_steps 5000  \
        --num_inference_diffusion_steps 100 \
        --beta_schedule squaredcos_improved_ddpm \
        --top_p 0.99 \
        --self_condition logits_mean \
        --self_condition_mix_before_weights \
        --is_causal false \
        --mask_padding_in_loss false \
        --dataset_name c4 --streaming --dataset_config_name en
