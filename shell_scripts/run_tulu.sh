# tulu command.
# WARNING: eval uses alpaca eval. this costs $$.

CMD="
python -m sdlm.run_tulu \
    --dataset_name allenai/tulu-v2-sft-mixture \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --do_train \
    --do_eval \
    --num_train_epochs 2 \
    --report_to tensorboard \
    --overwrite_output_dir \
    --max_seq_length 1024 \
    --simplex_value 5 \
    --num_diffusion_steps 5000 \
    --lr_scheduler_type cosine \
    --learning_rate 1e-5 \
    --pad_to_max_length \
    --beta_schedule squaredcos_improved_ddpm \
    --top_p 0.99 \
    --warmup_ratio 0.03 \
    --logging_steps 50 \
    --save_total_limit 1 \
    --save_strategy steps \
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
"

if [ ! -z "${BEAKER}" ]; then
    gantry run -y -n sni_mistral_dolma_adapt -t sni_mistral_dolma_adapt --allow-dirty \
        --workspace ai2/tess2 \
        --nfs \
        --gpus 1 \
        --priority normal \
        --budget ai2/allennlp \
        --cluster ai2/allennlp-cirrascale \
        --env 'HF_HOME=/net/nfs.cirrascale/allennlp/jaket/.hf' \
        --env 'PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python' \
        --venv 'base' \
        --pip requirements.txt \
        -- ${CMD} \
        --model_name_or_path /net/nfs.cirrascale/allennlp/jaket/simplex-diffusion/outputs/dolma_mistral/checkpoint-22000 \
        --evaluation_strategy epoch \
        --eval_steps 200 \
        --save_steps 400 \
        --max_eval_samples 1000 \
        --gradient_accumulation_steps 16 \
        --num_inference_diffusion_steps 50 100 200 \
        --beaker \
        --output_dir /results
else
    ${CMD} \
        --model_name_or_path /net/nfs.cirrascale/allennlp/jaket/simplex-diffusion/outputs/dolma_mistral/checkpoint-22000 \
        --evaluation_strategy steps \
        --eval_steps 1 \
        --save_steps 5 \
        --max_eval_samples 16 \
        --gradient_accumulation_steps 1 \
        --num_inference_diffusion_steps 10 \
        --output_dir outputs/test
fi

# using roberta (tess setup)
# gantry run -y -n tess_self_cond_tulu2_test -t tess_self_cond_tulu2_test --allow-dirty \
#     --workspace ai2/tess2 \
#     --nfs \
#     --gpus 1 \
#     --budget ai2/allennlp \
#     --priority normal \
#     --cluster ai2/allennlp-cirrascale \
#     --env 'HF_HOME=/net/nfs.cirrascale/allennlp/hamishi/.hf' \
#     --env 'PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python' \
#     --env 'IS_ALPACA_EVAL_2=False' \
#     --env-secret OPENAI_API_KEY=OPENAI_API_KEY \
#     --beaker-image 'ai2/pytorch2.0.0-cuda11.8-python3.10' \
#     --venv 'base' \
#     --pip requirements.txt \
#     -- python -m sdlm.run_tulu \
#     --model_name_or_path roberta-base \
#     --dataset_name allenai/tulu-v2-sft-mixture \
#     --output_dir /results \
#     --do_train \
#     --do_eval \
#     --max_seq_length 512 \
#     --per_device_train_batch_size 16 \
#     --skip_special_tokens False \
#     --per_device_eval_batch_size 16 \
#     --evaluation_strategy epoch \
#     --save_strategy steps \
#     --report_to tensorboard \
#     --overwrite_output_dir \
#     --pad_to_max_length \
#     --simplex_value 5 \
#     --num_train_epochs 2 \
#     --num_diffusion_steps 5000 \
#     --num_inference_diffusion_steps 100 \
#     --conditional_generation seq2seq \
#     --learning_rate 3e-5 \
#     --gradient_accumulation_steps 4 \
#     --lr_scheduler_type cosine \
#     --beta_schedule squaredcos_improved_ddpm \
#     --top_p 0.99 \
#     --warmup_ratio 0.03 \
#     --logging_steps 50 \
#     --save_total_limit 1 \
#     --max_eval_samples 1000 \
#     --self_condition "logits_mean" \
#     --self_condition_mix_before_weights

# using mistral :)
# gantry run -y -n mistral_1e5_self_cond_tulu2_test -t mistral_1e5_self_cond_tulu2_test --allow-dirty \
#     --workspace ai2/tess2 \
#     --nfs \
#     --gpus 1 \
#     --budget ai2/allennlp \
#     --priority normal \
#     --cluster ai2/allennlp-cirrascale \
#     --env 'HF_HOME=/net/nfs.cirrascale/allennlp/hamishi/.hf' \
#     --env 'PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python' \
#     --env 'IS_ALPACA_EVAL_2=False' \
#     --env-secret OPENAI_API_KEY=OPENAI_API_KEY \
#     --beaker-image 'ai2/pytorch2.0.0-cuda11.8-python3.10' \
#     --venv 'base' \
#     --pip requirements.txt \
#     -- python -m sdlm.run_tulu \
#     --model_name_or_path mistralai/Mistral-7B-v0.1 \
#     --dataset_name allenai/tulu-v2-sft-mixture \
#     --output_dir /results \
#     --do_train \
#     --do_eval \
#     --max_seq_length 1024 \
#     --per_device_train_batch_size 4 \
#     --skip_special_tokens False \
#     --per_device_eval_batch_size 4 \
#     --evaluation_strategy epoch \
#     --save_strategy steps \
#     --report_to tensorboard \
#     --overwrite_output_dir \
#     --pad_to_max_length \
#     --simplex_value 5 \
#     --num_train_epochs 2 \
#     --num_diffusion_steps 5000 \
#     --num_inference_diffusion_steps 100 \
#     --conditional_generation seq2seq \
#     --learning_rate 1e-5 \
#     --gradient_accumulation_steps 16 \
#     --lr_scheduler_type cosine \
#     --beta_schedule squaredcos_improved_ddpm \
#     --top_p 0.99 \
#     --warmup_ratio 0.03 \
#     --logging_steps 50 \
#     --save_total_limit 1 \
#     --max_eval_samples 1000 \
#     --self_condition "logits_mean" \
#     --self_condition_mix_before_weights \
#     --bf16 \
#     --optim adamw_torch_fused \
#     --gradient_checkpointing \
#     --use_flash_attention2 \
#     --save_safetensors true \
#     --is_causal false
