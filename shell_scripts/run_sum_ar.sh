CMD="
python -m sdlm.run_sum_ar \
    --model_name_or_path mistralai/Mistral-7B-v0.1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --do_train \
    --do_eval \
    --evaluation_strategy steps \
    --report_to tensorboard \
    --max_seq_length 1167 \
    --max_source_length 1024 \
    --max_target_length 142 \
    --lr_scheduler_type cosine \
    --learning_rate 1e-5 \
    --pad_to_max_length \
    --weight_decay 0.0 \
    --top_p 0.99 \
    --max_steps 120000 \
    --warmup_steps 2000 \
    --logging_steps 50 \
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
    --generation_max_length 1167 \
    --generation_num_beams 1 \
    --num_diffusion_steps 0 \
    --tokenizer_padding_side "left" \
"

if [ ! -z "${BEAKER}" ]; then
    gantry run -y -n cnndm_mistral_ar -t cnndm_mistral_ar --allow-dirty \
        --workspace ai2/tess2 \
        --nfs \
        --gpus 1 \
        --priority normal \
        --budget ai2/allennlp \
        --cluster ai2/allennlp-cirrascale \
        --env 'HF_HOME=/net/nfs.cirrascale/allennlp/jaket/.hf' \
        --env 'PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python' \
        --beaker-image 'ai2/pytorch2.0.0-cuda11.8-python3.10' \
        --venv 'base' \
        --pip requirements.txt \
        -- ${CMD} \
        --eval_steps 500 \
        --save_steps 1000 \
        --max_eval_samples 96 \
        --gradient_accumulation_steps 4 \
        --beaker \
        --output_dir /results
else
    ${CMD} \
        --eval_steps 1 \
        --save_steps 5 \
        --max_eval_samples 16 \
        --gradient_accumulation_steps 1 \
        --output_dir outputs/test
fi
