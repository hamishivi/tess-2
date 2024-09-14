CMD="
accelerate launch -m --mixed_precision bf16 sdlm.train_sentiment_model \
    --dataset_name cardiffnlp/tweet_eval \
    --dataset_config_name sentiment \
    --num_train_epochs 2 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --remove_unused_columns=False \
    --warmup_ratio 0.03 \
    --learning_rate=2e-5 \
    --logging_steps=50 \
    --save_total_limit 1 \
    --max_seq_length=512 \
    --gradient_checkpointing \
    --bf16 \
    --do_train \
    --do_eval \
    --optim adamw_torch_fused \
    --model_revision 26bca36bde8333b5d7f72e9ed20ccda6a618af24 \
    --shuffle_train_dataset \
    --metric_name accuracy \
    --text_column_name "text" \
    --text_column_delimiter "\n" \
    --label_column_name label \
    --overwrite_output_dir \
"

# on beaker, load from niklas' trained mistral model.
if [ ! -z "${BEAKER}" ]; then
    gantry run -y -n mistral_train_sentiment_classifier -t mistral_train_sentiment_classifier --allow-dirty \
        --workspace ai2/tess2 \
        --gpus 1 \
        --priority normal \
        --preemptible \
        --budget ai2/allennlp \
        --cluster ai2/pluto-cirrascale \
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
        --evaluation_strategy="epoch" \
        --gradient_accumulation_steps 32 \
        --output_dir /results
else
    ${CMD} \
        --model_name_or_path mistralai/Mistral-7B-v0.1 \
        --evaluation_strategy="steps" \
        --eval_steps 100 \
        --eval_steps 100 \
        --save_steps 100 \
        --gradient_accumulation_steps 32 \
        --output_dir outputs/test
fi
