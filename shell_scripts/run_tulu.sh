# tulu command.
# WARNING: eval uses alpaca eval. this costs $$.

model_path=$1

CMD="
accelerate launch
    --mixed_precision bf16 -m sdlm.run_tulu \
    --dataset_name allenai/tulu-v2-sft-mixture \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --evaluation_strategy epoch \
    --do_train \
    --do_eval \
    --num_train_epochs 3 \
    --report_to tensorboard \
    --max_seq_length 2048 \
    --simplex_value 5 \
    --num_diffusion_steps 5000 \
    --lr_scheduler_type cosine \
    --learning_rate 1e-5 \
    --pad_to_max_length \
    --beta_schedule squaredcos_improved_ddpm \
    --top_p 0.99 \
    --warmup_ratio 0.03 \
    --logging_steps 50 \
    --save_total_limit 2 \
    --save_strategy epoch \
    --conditional_generation seq2seq \
    --self_condition "logits_mean" \
    --self_condition_mix_before_weights \
    --bf16 \
    --optim adamw_torch_fused \
    --gradient_checkpointing \
    --use_flash_attention2 \
    --is_causal false
    --line_by_line true \
    --mask_padding_in_loss false \
    --skip_special_tokens false \
    --fsdp auto_wrap \
    --fsdp_transformer_layer_cls_to_wrap MistralDecoderLayer \
    --preprocessing_num_workers 16 \
    --model_revision 26bca36bde8333b5d7f72e9ed20ccda6a618af24 \
    --is_tulu_pair false \
    --is_tulu_multiturn false \
    --is_tulu_sliding_window_multiturn false \
    --eval_dataset_name alpaca_eval
"

if [ ! -z "${BEAKER}" ]; then
    gantry run -y -n tulu_v3_mistral -t tulu_v3_mistral --allow-dirty \
        --workspace ai2/tess2 \
        --gpus 7 \
        --priority normal \
        --budget ai2/allennlp \
        --preemptible \
        --no-nfs \
        --cluster ai2/allennlp-cirrascale \
        --cluster ai2/saturn-cirrascale \
        --cluster ai2/pluto-cirrascale \
        --cluster ai2/jupiter-cirrascale-2 \
        --env 'PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python' \
        --env 'IS_ALPACA_EVAL_2=False' \
        --env-secret OPENAI_API_KEY=OPENAI_API_KEY \
        --env-secret HF_TOKEN=HF_TOKEN \
        --beaker-image 'ai2/pytorch2.0.0-cuda11.8-python3.10' \
        --venv 'base' \
        --pip requirements.txt \
        -- ${CMD} \
        --model_name_or_path ${model_path} \
        --max_eval_samples 1000 \
        --gradient_accumulation_steps 8 \
        --num_inference_diffusion_steps 100 \
        --overwrite_output_dir false \
        --beaker \
        --output_dir /results
else
    ${CMD} \
        --model_name_or_path ${model_path} \
        --max_eval_samples 1000 \
        --gradient_accumulation_steps 8 \
        --num_inference_diffusion_steps 100 \
        --output_dir instruction_tuned_model \
        --overwrite_output_dir false
fi
