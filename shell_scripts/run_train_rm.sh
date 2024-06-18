CMD="
python -m sdlm.train_reward_model \
    --dataset_name argilla/ultrafeedback-binarized-preferences-cleaned \
     \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --remove_unused_columns=False \
    --gradient_checkpointing=True \
    --warmup_ratio 0.03 \
    --learning_rate=2e-5 \
    --report_to="tensorboard" \
    --logging_steps=50 \
    --save_total_limit 1 \
    --optim adamw_torch_fused \
    --evaluation_strategy="steps" \
    --max_length=512 \
    --gradient_checkpointing \
    --bf16 \
    --include_padding=False \
    --use_tulu_chat_template=True \
    --use_flash_attention2=True \
"

# on beaker, load from niklas' trained mistral model.
if [ ! -z "${BEAKER}" ]; then
    gantry run -y -n mistral_rm_train -t mistral_rm_train --allow-dirty \
        --workspace ai2/tess2 \
        --gpus 1 \
        --priority preemptible \
        --budget ai2/allennlp \
        --cluster ai2/jupiter-cirrascale-2 \
        --env 'HF_HOME=/net/nfs.cirrascale/allennlp/jaket/.hf' \
        --env 'PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python' \
        --dataset '01J0PF0NKZP7SD8TMRH2PD0NFK:/model' \
        --beaker-image 'ai2/pytorch2.0.0-cuda11.8-python3.10' \
        --env-secret HF_TOKEN=HF_TOKEN \
        --venv 'base' \
        --pip requirements.txt \
        -- ${CMD} \
        --model_name_or_path /model \
        --eval_steps 200 \
        --save_steps 400 \
        --gradient_accumulation_steps 128 \
        --output_dir /results
else
    ${CMD} \
        --model_name_or_path mistralai/Mistral-7B-v0.1 \
        --eval_steps 1 \
        --eval_steps 5 \
        --save_steps 5 \
        --gradient_accumulation_steps 1 \
        --output_dir outputs/test
fi
