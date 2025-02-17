# tulu command for eval.

run_name=$1
checkpoint_mount=$2
max_eval_samples=$3

CMD="
accelerate launch
    --mixed_precision bf16 -m sdlm.run_tulu_ar \
    --dataset_name allenai/tulu-v2-sft-mixture \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 16 \
    --evaluation_strategy steps \
    --do_eval \
    --num_train_epochs 2 \
    --report_to tensorboard \
    --max_seq_length 2048 \
    --simplex_value 5 \
    --num_diffusion_steps 1 \
    --lr_scheduler_type cosine \
    --learning_rate 2e-5 \
    --pad_to_max_length \
    --beta_schedule squaredcos_improved_ddpm \
    --top_p 0.99 \
    --warmup_ratio 0.03 \
    --logging_steps 50 \
    --save_total_limit 2 \
    --save_strategy steps \
    --conditional_generation seq2seq \
    --self_condition "logits_mean" \
    --self_condition_mix_before_weights \
    --bf16 \
    --optim adamw_torch_fused \
    --gradient_checkpointing \
    --use_flash_attention2 \
    --line_by_line true \
    --num_diffusion_steps 0 \
    --tokenizer_padding_side "left" \
    --mask_padding_in_loss false \
    --skip_special_tokens false \
    --eval_dataset_name squad
"

# for ai2/jupiter-cirrascale-2 cluster
if [ ! -z "${BEAKER}" ]; then
    gantry run -y -n $run_name -t $run_name --allow-dirty \
        --workspace ai2/tess2 \
        --gpus 8 \
        --priority normal \
        --budget ai2/allennlp \
        --preemptible \
        --no-nfs \
        --cluster ai2/jupiter-cirrascale-2 \
        --env 'HF_HOME=/net/weka/reviz/jaket/.hf' \
        --env 'PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python' \
        --env 'IS_ALPACA_EVAL_2=False' \
        --env-secret OPENAI_API_KEY=OPENAI_API_KEY \
        --beaker-image 'ai2/pytorch2.0.0-cuda11.8-python3.10' \
        --dataset "${checkpoint_mount}:/model" \
        --venv 'base' \
        --pip requirements.txt \
        -- ${CMD} \
        --model_name_or_path /model \
        --eval_steps 1000 \
        --save_steps 1000 \
        --max_eval_samples ${max_eval_samples} \
        --gradient_accumulation_steps 1 \
        --num_inference_diffusion_steps 100 250 \
        --overwrite_output_dir false \
        --beaker \
        --output_dir /results
else
    ${CMD} \
        --model_name_or_path ${checkpoint_mount} \
        --eval_steps 3 \
        --save_steps 5 \
        --max_eval_samples ${max_eval_samples} \
        --gradient_accumulation_steps 1 \
        --num_inference_diffusion_steps 100 \
        --output_dir outputs/test \
        --overwrite_output_dir true
fi