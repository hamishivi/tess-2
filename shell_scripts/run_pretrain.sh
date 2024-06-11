CMD="
accelerate launch
    --mixed_precision bf16 -m sdlm.run_pretrain \
    --model_name_or_path mistralai/Mistral-7B-v0.1 \
    --per_device_train_batch_size 2  \
    --per_device_eval_batch_size 2 \
    --do_train \
    --do_eval \
    --log_level info \
    --evaluation_strategy steps \
    --report_to tensorboard \
    --overwrite_output_dir false \
    --max_seq_length 2048 \
    --min_eval_seq_length 1050 \
    --simplex_value 5 \
    --num_diffusion_steps 5000  \
    --lr_scheduler_type cosine \
    --learning_rate 1e-5 \
    --pad_to_max_length \
    --beta_schedule squaredcos_improved_ddpm \
    --weight_decay 0.01 \
    --top_p 0.99 \
    --max_steps 200000 \
    --warmup_ratio 0.025 \
    --logging_steps 50 \
    --save_total_limit 1 \
    --conditional_generation ul2 \
    --self_condition "logits_mean" \
    --self_condition_mix_before_weights \
    --dataset_name NousResearch/dolma-v1_7-305B --streaming \
    --bf16 \
    --optim adamw_torch_fused \
    --gradient_checkpointing \
    --use_flash_attention2 \
    --is_causal false \
    --line_by_line true \
    --mask_padding_in_loss false \
    --ddp_find_unused_parameters false \
    --without_compute_metrics true \
"

if [ ! -z "${BEAKER}" ]; then
    gantry run -y -n dolma_mistral_4096_h100x8 -t dolma_mistral_4096_h100x8 --allow-dirty \
        --workspace ai2/tess2 \
        --gpus 8 \
        --priority normal \
        --budget ai2/allennlp \
        --preemptible \
        --no-nfs \
        --cluster ai2/jupiter-cirrascale-2 \
        --env 'HF_HOME=/net/weka/reviz/jaket/.hf' \
        --env 'PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python' \
        --env-secret HF_TOKEN=HF_TOKEN \
        --beaker-image 'ai2/pytorch2.0.0-cuda11.8-python3.10' \
        --venv 'base' \
        --pip requirements.txt \
        -- ${CMD} \
        --eval_steps 1000 \
        --save_steps 1000 \
        --max_eval_samples 200 \
        --gradient_accumulation_steps 8 \
        --num_inference_diffusion_steps 100 \
        --beaker \
        --output_dir /results
else
    ${CMD} \
        --eval_steps 10 \
        --save_steps 50 \
        --max_eval_samples 16 \
        --gradient_accumulation_steps 1 \
        --num_inference_diffusion_steps 10 \
        --output_dir outputs/test
fi
