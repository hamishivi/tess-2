EXP_NAME="prefix_uncond_100k_c4_roberta_base_shorter_gen"

# gantry run -y -n $EXP_NAME -t $EXP_NAME --allow-dirty \
#     --workspace ai2/tess2 \
#     --nfs \
#     --gpus 1 \
#     --priority normal \
#     --cluster ai2/allennlp-cirrascale \
#     --env 'HF_HOME=/net/nfs.cirrascale/allennlp/hamishi/.hf' \
#     --env 'PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python' \
#     --beaker-image 'ai2/pytorch2.0.0-cuda11.8-python3.10' \
#     --venv 'base' \
#     --pip requirements.txt \
#     -- python sdlm/run_mlm.py \
#         --model_name_or_path roberta-base \
#         --per_device_train_batch_size 48  \
#         --per_device_eval_batch_size 48 \
#         --do_train \
#         --do_eval \
#         --output_dir /results \
#         --evaluation_strategy steps \
#         --eval_steps 1000 \
#         --report_to tensorboard \
#         --max_seq_length 25  \
#         --max_eval_samples 512 \
#         --simplex_value 5 \
#         --num_diffusion_steps 5000  \
#         --num_inference_diffusion_steps 1 10 100 500 \
#         --lr_scheduler_type cosine \
#         --learning_rate 1e-4 \
#         --pad_to_max_length \
#         --beta_schedule squaredcos_improved_ddpm \
#         --weight_decay 0.01 \
#         --top_p 0.99 \
#         --max_steps 100000 \
#         --gradient_accumulation_steps 4 \
#         --warmup_steps 2000 \
#         --logging_steps 50 \
#         --save_steps 1000 \
#         --self_condition "logits_mean" \
#         --self_condition_mix_before_weights \
#         --beaker \
#         --conditional_generation prefix_with_unconditional \
#         --dataset_name c4 --streaming --dataset_config_name en


EXP_NAME="ul2_orig_100k_c4_roberta_base_long_context_short_gen"

gantry run -y -n $EXP_NAME -t $EXP_NAME --allow-dirty \
    --workspace ai2/tess2 \
    --nfs \
    --gpus 1 \
    --priority normal \
    --cluster ai2/allennlp-cirrascale \
    --env 'HF_HOME=/net/nfs.cirrascale/allennlp/hamishi/.hf' \
    --env 'PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python' \
    --beaker-image 'ai2/pytorch2.0.0-cuda11.8-python3.10' \
    --venv 'base' \
    --pip requirements.txt \
    -- python sdlm/run_mlm.py \
        --model_name_or_path roberta-base \
        --per_device_train_batch_size 48  \
        --per_device_eval_batch_size 48 \
        --do_train \
        --do_eval \
        --output_dir /results \
        --evaluation_strategy steps \
        --eval_steps 1000 \
        --report_to tensorboard \
        --max_seq_length 256  \
        --eval_context_size 231 \
        --max_eval_samples 512 \
        --simplex_value 5 \
        --num_diffusion_steps 5000  \
        --num_inference_diffusion_steps 1 10 100 500 \
        --lr_scheduler_type cosine \
        --learning_rate 1e-4 \
        --pad_to_max_length \
        --beta_schedule squaredcos_improved_ddpm \
        --weight_decay 0.01 \
        --top_p 0.99 \
        --max_steps 100000 \
        --gradient_accumulation_steps 4 \
        --warmup_steps 2000 \
        --logging_steps 50 \
        --save_steps 1000 \
        --conditional_generation ul2 \
        --self_condition "logits_mean" \
        --beaker \
        --self_condition_mix_before_weights \
        --dataset_name c4 --streaming --dataset_config_name en


# EXP_NAME="ul2_orig_100k_c4_roberta_scratch_base_self_cond_big_eval"

# gantry run -y -n $EXP_NAME -t $EXP_NAME --allow-dirty \
#     --workspace ai2/tess2 \
#     --nfs \
#     --gpus 1 \
#     --priority normal \
#     --cluster ai2/allennlp-cirrascale \
#     --env 'HF_HOME=/net/nfs.cirrascale/allennlp/hamishi/.hf' \
#     --env 'PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python' \
#     --beaker-image 'ai2/pytorch2.0.0-cuda11.8-python3.10' \
#     --venv 'base' \
#     --pip requirements.txt \
#     -- python sdlm/run_mlm.py \
#         --model_name_or_path roberta-base \
#         --per_device_train_batch_size 48  \
#         --per_device_eval_batch_size 48 \
#         --do_train \
#         --do_eval \
#         --output_dir /results \
#         --evaluation_strategy steps \
#         --eval_steps 1000 \
#         --report_to tensorboard \
#         --max_seq_length 256  \
#         --max_eval_samples 512 \
#         --simplex_value 5 \
#         --num_diffusion_steps 5000  \
#         --num_inference_diffusion_steps 1 10 100 500 \
#         --lr_scheduler_type cosine \
#         --learning_rate 1e-4 \
#         --pad_to_max_length \
#         --beta_schedule squaredcos_improved_ddpm \
#         --weight_decay 0.01 \
#         --top_p 0.99 \
#         --max_steps 100000 \
#         --gradient_accumulation_steps 4 \
#         --warmup_steps 2000 \
#         --logging_steps 50 \
#         --save_steps 1000 \
#         --conditional_generation ul2 \
#         --beaker \
#         --self_condition "logits_mean" \
#         --self_condition_mix_before_weights \
#         --dataset_name c4 --streaming --dataset_config_name en \
#         --from_scratch
