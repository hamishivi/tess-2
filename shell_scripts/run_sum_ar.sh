CMD="
python -m sdlm.run_sum_ar \
    --model_name_or_path mistralai/Mistral-7B-v0.1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --do_train \
    --do_eval \
    --output_dir /results \
    --evaluation_strategy steps \
    --eval_steps 1 \
    --report_to tensorboard \
    --max_seq_length 1166 \
    --max_source_length 1024 \
    --max_target_length 142 \
    --max_eval_samples 96 \
    --lr_scheduler_type cosine \
    --learning_rate 1e-5 \
    --pad_to_max_length \
    --weight_decay 0.0 \
    --top_p 0.99 \
    --max_steps 120000 \
    --gradient_accumulation_steps 4 \
    --warmup_steps 2000 \
    --logging_steps 50 \
    --save_steps 1000 \
    --save_total_limit 1 \
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
    --generation_max_length 1166 \
    --generation_num_beams 1 \
    --num_diffusion_steps 0 \
    --tokenizer_padding_side "left"
"

if [ -z "${BEAKER_EXP}" ]; then
        gantry run -y -n cnndm_mistral_ar -t cnndm_mistral_ar --allow-dirty \
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
