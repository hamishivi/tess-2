#!/bin/bash

model_name_or_path=$1
reward_model_name_or_path=$2
guidance_scale=$3
eval_name=$4

accelerate launch \
    --mixed_precision bf16 -m sdlm.run_tulu \
    --model_name_or_path $model_name_or_path \
    --dataset_name allenai/tulu-v2-sft-mixture \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --do_train false \
    --do_eval true \
    --load_states_in_eval_from_model_path false \
    --log_level info \
    --evaluation_strategy steps \
    --report_to tensorboard \
    --overwrite_output_dir \
    --max_seq_length 2048 \
    --min_eval_seq_length 512 \
    --simplex_value 5 \
    --num_diffusion_steps 5000  \
    --lr_scheduler_type cosine \
    --learning_rate 1e-5 \
    --pad_to_max_length \
    --beta_schedule squaredcos_improved_ddpm \
    --weight_decay 0.01 \
    --top_p 0.99 \
    --max_steps 100000 \
    --warmup_ratio 0.05 \
    --logging_steps 50 \
    --save_total_limit 1 \
    --conditional_generation ul2 \
    --self_condition "logits_mean" \
    --self_condition_mix_before_weights \
    --bf16 \
    --optim adamw_torch_fused \
    --gradient_checkpointing \
    --use_flash_attention2 \
    --is_causal false \
    --line_by_line true \
    --mask_padding_in_loss false \
    --ddp_find_unused_parameters false \
    --without_compute_metrics true \
    --classifier_model_name_or_path $reward_model_name_or_path \
    --guidance_scale ${guidance_scale} \
    --use_gumbel_softmax false \
    --do_hard_sample false \
    --eval_dataset_name $eval_name \
    --max_eval_samples 1500 \
    --num_inference_diffusion_steps 100 \
    --output_dir outputs/${model_name_or_path}_${guidance_scale}
