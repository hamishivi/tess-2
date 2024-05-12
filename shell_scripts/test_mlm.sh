python -m sdlm.run_mlm \
        --model_name_or_path outputs/llama/c4/try6_no_mask_pad_yes_lbl/checkpoint-5000 \
        --per_device_train_batch_size 16  \
        --per_device_eval_batch_size 16 \
        --do_train \
        --do_eval \
        --output_dir outputs/llama/c4/try6_no_mask_pad_yes_lbl \
        --evaluation_strategy steps \
        --eval_steps 100 \
        --report_to tensorboard \
        --max_seq_length 512  \
        --max_eval_samples 96 \
        --simplex_value 5 \
        --num_diffusion_steps 5000  \
        --num_inference_diffusion_steps 100 \
        --lr_scheduler_type cosine \
        --learning_rate 1e-5 \
        --pad_to_max_length \
        --beta_schedule squaredcos_improved_ddpm \
        --weight_decay 0.01 \
        --top_p 0.99 \
        --max_steps 50000 \
        --gradient_accumulation_steps 4 \
        --warmup_ratio 0.05 \
        --logging_steps 50 \
        --save_steps 1000 \
        --save_total_limit 3 \
        --conditional_generation ul2 \
        --self_condition "logits_mean" \
        --self_condition_mix_before_weights \
        --dataset_name c4 --streaming --dataset_config_name en \
        --bf16 \
        --optim adamw_torch_fused \
        --gradient_checkpointing \
        --use_flash_attention2 \
        --is_causal false \
        --line_by_line true \
        --eval_long_only true \
        --mask_padding_in_loss false
