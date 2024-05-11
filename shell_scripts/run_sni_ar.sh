CMD="
python -m sdlm.run_sni_ar \
    --model_name_or_path mistralai/Mistral-7B-v0.1 \
    --dataset_name sni \
    --output_dir /results \
    --do_train \
    --do_eval \
    --max_seq_length 1152 \
    --max_source_length 1024 \
    --max_target_length 128 \
    --skip_special_tokens true \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --evaluation_strategy steps \
    --eval_steps 256 \
    --save_strategy steps \
    --report_to tensorboard \
    --overwrite_output_dir \
    --pad_to_max_length \
    --num_train_epochs 4 \
    --conditional_generation seq2seq \
    --learning_rate 1e-5 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.03 \
    --logging_steps 50 \
    --save_total_limit 1 \
    --max_eval_samples 512 \
    --preprocessing_num_workers 16 \
    --bf16 \
    --optim adamw_torch_fused \
    --gradient_checkpointing \
    --use_flash_attention2 \
    --is_causal true \
    --mask_padding_in_loss false \
    --generation_max_length 1152 \
    --generation_num_beams 1 \
    --num_diffusion_steps 0 \
    --tokenizer_padding_side "left"
"

if [ -z "${GANTRY}" ]; then
    gantry run -y -n sni_mistral_ar -t sni_mistral_ar --allow-dirty \
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
        -- ${CMD}
else
    ${CMD}
fi
