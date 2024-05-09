# # roberta diffusion sni baseline
# gantry run -y -n sni_roberta_1e-5 -t sni_roberta_1e-5 --allow-dirty \
#     --workspace ai2/tess2 \
#     --nfs \
#     --gpus 1 \
#     --priority normal \
#     --budget ai2/allennlp \
#     --cluster ai2/allennlp-cirrascale \
#     --env 'HF_HOME=/net/nfs.cirrascale/allennlp/jaket/.hf' \
#     --env 'PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python' \
#     --venv 'base' \
#     --pip requirements.txt \
#     -- python -m sdlm.run_glue \
#         --model_name_or_path roberta-base \
#         --dataset_name sni \
#         --output_dir /results \
#         --do_train \
#         --do_eval \
#         --max_seq_length 512 \
#         --per_device_train_batch_size 16 \
#         --skip_special_tokens True \
#         --per_device_eval_batch_size 16 \
#         --evaluation_strategy steps \
#         --eval_steps 2048 \
#         --save_strategy steps \
#         --report_to tensorboard \
#         --overwrite_output_dir \
#         --pad_to_max_length \
#         --simplex_value 5 \
#         --num_train_epochs 2 \
#         --num_diffusion_steps 5000 \
#         --num_inference_diffusion_steps 100 \
#         --conditional_generation seq2seq \
#         --learning_rate 1e-5 \
#         --gradient_accumulation_steps 8 \
#         --lr_scheduler_type cosine \
#         --beta_schedule squaredcos_improved_ddpm \
#         --top_p 0.99 \
#         --warmup_ratio 0.03 \
#         --logging_steps 50 \
#         --save_total_limit 1 \
#         --max_eval_samples 1000 \
#         --preprocessing_num_workers 16 \
#         --self_condition "logits_mean" \
#         --self_condition_mix_before_weights

# mistral diffusion sni
gantry run -y -n sni_100_mistral -t sni_100_mistral --allow-dirty \
    --workspace ai2/tess2 \
    --nfs \
    --gpus 1 \
    --priority normal \
    --budget ai2/allennlp \
    --cluster ai2/allennlp-cirrascale \
    --env 'HF_HOME=/net/nfs.cirrascale/allennlp/jaket/.hf' \
    --env 'PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python' \
    --venv 'base' \
    --pip requirements.txt \
    -- python -m sdlm.run_glue \
        --model_name_or_path mistralai/Mistral-7B-v0.1 \
        --dataset_name sni \
        --output_dir /results \
        --do_train \
        --do_eval \
        --max_seq_length 512 \
        --per_device_train_batch_size 16 \
        --skip_special_tokens true \
        --per_device_eval_batch_size 16 \
        --evaluation_strategy steps \
        --eval_steps 2048 \
        --save_strategy steps \
        --report_to tensorboard \
        --overwrite_output_dir \
        --pad_to_max_length \
        --simplex_value 5 \
        --num_train_epochs 2 \
        --num_diffusion_steps 5000 \
        --num_inference_diffusion_steps 100 500 1000 \
        --conditional_generation seq2seq \
        --learning_rate 1e-5 \
        --gradient_accumulation_steps 4 \
        --lr_scheduler_type cosine \
        --beta_schedule squaredcos_improved_ddpm \
        --top_p 0.99 \
        --warmup_ratio 0.03 \
        --logging_steps 50 \
        --save_total_limit 1 \
        --max_eval_samples 512 \
        --preprocessing_num_workers 16 \
        --self_condition "logits_mean" \
        --self_condition_mix_before_weights \
        --bf16 \
        --optim adamw_torch_fused \
        --gradient_checkpointing \
        --use_flash_attention2 \
        --is_causal false \
        --mask_padding_in_loss false
