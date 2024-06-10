CMD="
accelerate launch
    --mixed_precision bf16 -m sdlm.run_pretrain \
    --model_name_or_path mistralai/Mistral-7B-v0.1 \
    --per_device_train_batch_size 1  \
    --per_device_eval_batch_size 1 \
    --do_train \
    --do_eval \
    --log_level info \
    --evaluation_strategy steps \
    --report_to tensorboard \
    --overwrite_output_dir \
    --max_seq_length 4096  \
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
    --save_total_limit 2 \
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
    gantry run -y -n dolma_mistral_long -t dolma_mistral_long --allow-dirty \
        --workspace ai2/tess2 \
        --nfs \
        --gpus 4 \
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
        --save_steps 1000 \
        --max_eval_samples 512 \
        --gradient_accumulation_steps 4 \
        --num_inference_diffusion_steps 100 200 \
        --beaker \
        --output_dir /results
else
    ${CMD} \
        --eval_steps 10 \
        --save_steps 50 \
        --max_eval_samples 16 \
        --gradient_accumulation_steps 4 \
        --num_inference_diffusion_steps 10 \
        --output_dir outputs/test
fi
