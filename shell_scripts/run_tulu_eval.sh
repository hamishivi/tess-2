# tulu eval command for number of diffusion steps.
# WARNING: eval uses alpaca eval. this costs $$.

checkpoint_mount="01J16DWYN44SEZT1F70PD13MYN"

CMD="
accelerate launch
    --mixed_precision bf16 -m sdlm.run_tulu \
    --dataset_name allenai/tulu-v2-sft-mixture \
    --per_device_eval_batch_size 8 \
    --evaluation_strategy epoch \
    --do_train false \
    --do_eval true \
    --num_train_epochs 0 \
    --report_to tensorboard \
    --max_seq_length 512 \
    --simplex_value 5 \
    --num_diffusion_steps 5000 \
    --pad_to_max_length \
    --beta_schedule squaredcos_improved_ddpm \
    --top_p 0.99 \
    --logging_steps 50 \
    --conditional_generation seq2seq \
    --self_condition "logits_mean" \
    --self_condition_mix_before_weights \
    --bf16 \
    --use_flash_attention2 \
    --is_causal false
    --line_by_line true \
    --mask_padding_in_loss false \
    --skip_special_tokens false \
"

# for ai2/allennlp-cirrascale cluster
# if [ ! -z "${BEAKER}" ]; then
#     gantry run -y -n tulu_mistral_512_constant_250 -t tulu_mistral_512_constant_250 --allow-dirty \
#         --workspace ai2/tess2 \
#         --nfs \
#         --gpus 8 \
#         --priority normal \
#         --budget ai2/allennlp \
#         --cluster ai2/allennlp-cirrascale \
#         --env 'HF_HOME=/net/nfs.cirrascale/allennlp/jaket/.hf' \
#         --env 'PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python' \
#         --env 'IS_ALPACA_EVAL_2=False' \
#         --env-secret OPENAI_API_KEY=OPENAI_API_KEY \
#         --beaker-image 'ai2/pytorch2.0.0-cuda11.8-python3.10' \
#         --dataset "${checkpoint_mount}:/model" \
#         --venv 'base' \
#         --pip requirements.txt \
#         -- ${CMD} \
#         --model_name_or_path /model \
#         --max_eval_samples 1000 \
#         --num_inference_diffusion_steps 250 \
#         --overwrite_output_dir false \
#         --beaker \
#         --output_dir /results
# else
#     ${CMD} \
#         --model_name_or_path mistralai/Mistral-7B-v0.1 \
#         --max_eval_samples 16 \
#         --num_inference_diffusion_steps 10 \
#         --output_dir outputs/test \
#         --overwrite_output_dir true
# fi

# for ai2/jupiter-cirrascale-2 cluster
if [ ! -z "${BEAKER}" ]; then
    gantry run -y -n tulu_mistral_512_constant_250 -t tulu_mistral_512_constant_250 --allow-dirty \
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
        --max_eval_samples 1000 \
        --num_inference_diffusion_steps 75 \
        --overwrite_output_dir false \
        --beaker \
        --output_dir /results
else
    ${CMD} \
        --model_name_or_path mistralai/Mistral-7B-v0.1 \
        --max_eval_samples 16 \
        --num_inference_diffusion_steps 10 \
        --output_dir outputs/test \
        --overwrite_output_dir true
fi
