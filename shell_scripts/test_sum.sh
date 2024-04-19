# NOTE: llama finetune w/o adaptation (causal)
gantry run -y -n mistral-cnn-dm -t mistral-cnn-dm --budget ai2/allennlp --allow-dirty \
    --workspace ai2/tess2 \
    --nfs \
    --gpus 1 \
    --priority high \
    --cluster ai2/allennlp-cirrascale \
    --env 'HF_HOME=/net/nfs.cirrascale/allennlp/hamishi/.hf' \
    --env 'PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python' \
    --venv 'base' \
    --pip requirements.txt \
    -- python -m sdlm.run_summarization \
        --model_name_or_path mistralai/Mistral-7B-v0.1 \
        --per_device_train_batch_size 16  \
        --per_device_eval_batch_size 16 \
        --do_train \
        --do_eval \
        --output_dir /results \
        --evaluation_strategy steps \
        --eval_steps 100 \
        --report_to tensorboard \
        --max_seq_length 512  \
        --max_source_length 392 \
        --max_target_length 120 \
        --max_eval_samples 96 \
        --simplex_value 5 \
        --num_diffusion_steps 5000  \
        --num_inference_diffusion_steps 100 \
        --lr_scheduler_type cosine \
        --learning_rate 5e-6 \
        --pad_to_max_length \
        --beta_schedule squaredcos_improved_ddpm \
        --weight_decay 0.0 \
        --top_p 0.99 \
        --max_steps 120000 \
        --gradient_accumulation_steps 4 \
        --warmup_steps 2000 \
        --logging_steps 50 \
        --save_steps 2000 \
        --save_total_limit 3 \
        --conditional_generation "seq2seq" \
        --self_condition "logits_mean" \
        --self_condition_mix_before_weights \
        --dataset_name cnn_dailymail --dataset_config "3.0.0" \
        --overwrite_output_dir \
        --bf16 \
        --optim adamw_torch_fused \
        --gradient_checkpointing \
        --use_flash_attention2 \
        --save_safetensors true \
        --is_causal false \
        --mask_padding_in_loss false

# NOTE: attempted baseline
# python -m backups.baselines.run_summarization \
#         --model_name_or_path meta-llama/Llama-2-7b-hf \
#         --per_device_train_batch_size 16  \
#         --per_device_eval_batch_size 16 \
#         --do_train \
#         --do_eval \
#         --output_dir outputs/llama/finetune_cnn_dm \
#         --evaluation_strategy steps \
#         --eval_steps 100 \
#         --report_to tensorboard \
#         --max_source_length 392 \
#         --max_target_length 120 \
#         --max_eval_samples 48 \
#         --lr_scheduler_type cosine \
#         --learning_rate 3e-5 \
#         --pad_to_max_length \
#         --weight_decay 0.0 \
#         --top_p 0.99 \
#         --max_steps 120000 \
#         --gradient_accumulation_steps 4 \
#         --warmup_steps 2000 \
#         --logging_steps 50 \
#         --save_steps 1000 \
#         --save_total_limit 3 \
#         --dataset_name cnn_dailymail --dataset_config "3.0.0" \
#         --overwrite_output_dir \
#         --bf16 \
#         --optim adamw_torch_fused \
#         --gradient_checkpointing \
#         --save_safetensors true

# NOTE: tess v1 repro
# python -m sdlm.run_summarization \
#         --model_name_or_path roberta-base \
#         --do_train \
#         --do_eval \
#         --dataset_name "cnn_dailymail" \
#         --dataset_config "3.0.0" \
#         --output_dir "outputs/roberta/try6" \
#         --per_device_train_batch_size 24 \
#         --per_device_eval_batch_size 48 \
#         --overwrite_output_dir \
#         --report_to tensorboard \
#         --evaluation_strategy steps \
#         --eval_steps 100 \
#         --max_steps 120000 \
#         --max_eval_samples 96 \
#         --max_source_length 392 \
#         --max_target_length 120 \
#         --max_seq_length 512 \
#         --conditional_generation "seq2seq" \
#         --num_inference_diffusion_steps 1000 \
#         --simplex_value 5 \
#         --num_diffusion_steps 5000 \
#         --lr_scheduler_type linear \
#         --learning_rate 3e-5 \
#         --pad_to_max_length \
#         --beta_schedule squaredcos_improved_ddpm \
#         --weight_decay 0.0 \
#         --warmup_steps 2000 \
#         --logging_steps 50 \
#         --save_steps 20000 \
#         --save_total_limit 2 \
#         --self_condition "logits_mean" \
#         --self_condition_mix_before_weights true \
#         --gradient_accumulation_steps 2 \
#         --mask_padding_in_loss false
