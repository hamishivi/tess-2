python -m sdlm.run_sum_ar \
        --model_name_or_path meta-llama/Llama-2-7b-hf \
        --per_device_train_batch_size 16  \
        --per_device_eval_batch_size 16 \
        --do_train \
        --do_eval \
        --output_dir outputs/llama/cnn_dm/ar_baseline \
        --evaluation_strategy steps \
        --eval_steps 100 \
        --report_to tensorboard \
        --max_seq_length 512  \
        --max_source_length 391 \
        --max_target_length 120 \
        --max_eval_samples 96 \
        --lr_scheduler_type cosine \
        --learning_rate 3e-5 \
        --pad_to_max_length \
        --weight_decay 0.0 \
        --top_p 0.99 \
        --max_steps 120000 \
        --gradient_accumulation_steps 4 \
        --warmup_steps 2000 \
        --logging_steps 50 \
        --save_steps 1000 \
        --save_total_limit 2 \
        --conditional_generation "seq2seq" \
        --dataset_name cnn_dailymail --dataset_config "3.0.0" \
        --overwrite_output_dir \
        --bf16 \
        --optim adamw_torch_fused \
        --gradient_checkpointing \
        --use_flash_attention2 \
        --save_safetensors true \
        --is_causal true \
        --mask_padding_in_loss false \
        --generation_max_length 512 \
        --generation_num_beams 1 \
        --num_diffusion_steps 0 \
