CMD="
python -m sdlm.run_glue \
    --model_name_or_path mistralai/Mistral-7B-v0.1 \
    --dataset_name sni \
    --do_train \
    --do_eval \
    --max_seq_length 1024 \
    --max_source_length 896 \
    --max_target_length 128 \
    --skip_special_tokens true \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --evaluation_strategy steps \
    --save_strategy steps \
    --report_to tensorboard \
    --overwrite_output_dir \
    --pad_to_max_length \
    --simplex_value 5 \
    --num_train_epochs 2 \
    --num_diffusion_steps 5000 \
    --conditional_generation seq2seq \
    --learning_rate 1e-5 \
    --lr_scheduler_type cosine \
    --beta_schedule squaredcos_improved_ddpm \
    --top_p 0.99 \
    --warmup_ratio 0.03 \
    --logging_steps 50 \
    --save_total_limit 1 \
    --preprocessing_num_workers 16 \
    --self_condition "logits_mean" \
    --self_condition_mix_before_weights \
    --bf16 \
    --optim adamw_torch_fused \
    --gradient_checkpointing \
    --use_flash_attention2 \
    --is_causal false \
    --mask_padding_in_loss false \
"

if [ ! -z "${BEAKER}" ]; then
    gantry run -y -n sni_mistral -t sni_mistral --allow-dirty \
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
        --eval_steps 200 \
        --save_steps 400 \
        --max_eval_samples 512 \
        --gradient_accumulation_steps 4 \
        --num_inference_diffusion_steps 50 100 200 \
        --beaker \
        --output_dir /results
else
    ${CMD} \
        --eval_steps 1 \
        --save_steps 5 \
        --max_eval_samples 16 \
        --gradient_accumulation_steps 1 \
        --num_inference_diffusion_steps 10 \
        --output_dir outputs/test
fi
