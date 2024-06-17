# tulu command.
# WARNING: eval uses alpaca eval. this costs $$.

CMD="
python -m sdlm.run_tulu_ar \
    --dataset_name allenai/tulu-v2-sft-mixture \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --evaluation_strategy epoch \
    --do_train \
    --do_eval \
    --num_train_epochs 2 \
    --report_to tensorboard \
    --overwrite_output_dir \
    --max_seq_length 512 \
    --simplex_value 5 \
    --num_diffusion_steps 5000 \
    --lr_scheduler_type cosine \
    --learning_rate 2e-5 \
    --conditional_generation "seq2seq" \
    --is_causal true \
    --warmup_ratio 0.03 \
    --logging_steps 50 \
    --save_total_limit 1 \
    --pad_to_max_length \
    --bf16 \
    --optim adamw_torch_fused \
    --gradient_checkpointing \
    --use_flash_attention2 \
    --save_strategy steps \
    --gradient_checkpointing \
    --num_diffusion_steps 0 \
    --tokenizer_padding_side "left" \
"

# for ai2/jupiter-cirrascale-2 cluster
if [ ! -z "${BEAKER}" ]; then
    gantry run -y -n mistral_tulu_ar_baseline -t mistral_tulu_ar_baseline --allow-dirty \
        --workspace ai2/tess2 \
        --gpus 8 \
        --priority preemptible \
        --budget ai2/allennlp \
        --preemptible \
        --no-nfs \
        --cluster ai2/jupiter-cirrascale-2 \
        --env 'HF_HOME=/net/weka/reviz/jaket/.hf' \
        --env 'PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python' \
        --env 'IS_ALPACA_EVAL_2=False' \
        --env-secret OPENAI_API_KEY=OPENAI_API_KEY \
        --env-secret HF_TOKEN=HF_TOKEN \
        --beaker-image 'ai2/pytorch2.0.0-cuda11.8-python3.10' \
        --venv 'base' \
        --pip requirements.txt \
        -- ${CMD} \
        --model_name_or_path mistralai/Mistral-7B-v0.1 \
        --save_steps 1000 \
        --max_eval_samples 1000 \
        --gradient_accumulation_steps 2 \
        --output_dir /results
else
    ${CMD} \
        --model_name_or_path mistralai/Mistral-7B-v0.1 \
        --eval_steps 3 \
        --save_steps 5 \
        --max_eval_samples 16 \
        --gradient_accumulation_steps 1 \
        --num_inference_diffusion_steps 10 \
        --output_dir outputs/test
fi
