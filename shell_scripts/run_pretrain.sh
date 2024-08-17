PYTHON_CMD="
accelerate launch
    --mixed_precision bf16 -m sdlm.run_pretrain \
    --per_device_train_batch_size 16  \
    --per_device_eval_batch_size 16 \
    --do_train \
    --do_eval \
    --log_level info \
    --evaluation_strategy steps \
    --report_to tensorboard \
    --max_seq_length 1024 \
    --simplex_value 5 \
    --num_diffusion_steps 5000  \
    --lr_scheduler_type constant_with_warmup \
    --learning_rate 1e-5 \
    --pad_to_max_length \
    --beta_schedule squaredcos_improved_ddpm \
    --top_p 0.99 \
    --max_steps 10000000 \
    --warmup_ratio 0.0125 \
    --logging_steps 50 \
    --save_total_limit 1 \
    --conditional_generation ul2 \
    --self_condition logits_mean \
    --self_condition_mix_before_weights \
    --streaming \
    --bf16 \
    --optim adamw_torch_fused \
    --gradient_checkpointing \
    --use_flash_attention2 \
    --is_causal false \
    --mask_padding_in_loss false \
    --ddp_find_unused_parameters false \
    --without_compute_metrics true \
    --dataloader_num_workers 8 \
    --remove_unused_columns false \
    --dispatch_batches false \
    --shuffle true \
    --preprocessing_num_workers 16 \
    --model_revision 26bca36bde8333b5d7f72e9ed20ccda6a618af24 \
    --line_by_line false \
"
    # --min_sample_seq_length 650 \  # set to filter by length
    # --use_model cdcd \  # cdcd


GANTRY_CMD="
gantry run -y -n dolma_mistral_1k -t dolma_mistral_1k --allow-dirty \
    --workspace ai2/tess2 \
    --gpus 8 \
    --priority normal \
    --budget ai2/allennlp \
    --preemptible \
    --env PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
    --env-secret HF_TOKEN=HF_TOKEN \
    --beaker-image ai2/pytorch2.0.0-cuda11.8-python3.10 \
    --venv base \
    --pip requirements.txt \
    -- ${PYTHON_CMD} \
    --model_name_or_path mistralai/Mistral-7B-v0.1 \
    --eval_steps 2000 \
    --save_steps 2000 \
    --max_eval_samples 200 \
    --gradient_accumulation_steps 1 \
    --num_inference_diffusion_steps 100 \
    --overwrite_output_dir false \
    --beaker \
    --output_dir /results \
"


if [ ! -z "${BEAKER}" ]; then
    if [ ! -z "${WEKA}" ]; then
        ${GANTRY_CMD} \
            --no-nfs \
            --cluster ai2/jupiter-cirrascale-2 \
            --weka oe-data-default:/data/input
    else
        ${GANTRY_CMD} \
            --cluster ai2/allennlp-cirrascale \
            --cluster ai2/pluto-cirrascale \
            --cluster ai2/general-cirrascale-a100-80g-ib
    fi
else
    ${PYTHON_CMD} \
        --model_name_or_path mistralai/Mistral-7B-v0.1 \
        --eval_steps 10 \
        --save_steps 50 \
        --max_eval_samples 16 \
        --gradient_accumulation_steps 1 \
        --num_inference_diffusion_steps 10 \
        --output_dir outputs/test \
        --overwrite_output_dir true
fi
