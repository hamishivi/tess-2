# this was for prefix, but didnt work as well as ul2.

# EXP_NAME="ul2_orig_100k_c4_roberta_base_pretrain_50k_fixed"

# gantry run -y -n $EXP_NAME -t $EXP_NAME --allow-dirty \
#     --workspace ai2/tess2 \
#     --nfs \
#     --gpus 1 \
#     --priority high \
#     --cluster ai2/allennlp-cirrascale \
#     --env 'HF_HOME=/net/nfs.cirrascale/allennlp/hamishi/.hf' \
#     --env 'PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python' \
#     --beaker-image 'ai2/pytorch2.0.0-cuda11.8-python3.10' \
#     --venv 'base' \
#     --pip requirements.txt \
#     -- python -m sdlm.run_mlm \
#         --model_name_or_path roberta-base \
#         --per_device_train_batch_size 24  \
#         --per_device_eval_batch_size 24 \
#         --do_train \
#         --do_eval \
#         --output_dir /results \
#         --evaluation_strategy steps \
#         --eval_steps 1000 \
#         --report_to tensorboard \
#         --run_name $EXP_NAME \
#         --max_seq_length 512  \
#         --max_eval_samples 512 \
#         --simplex_value 5 \
#         --num_diffusion_steps 5000  \
#         --num_inference_diffusion_steps 10 100 \
#         --lr_scheduler_type cosine \
#         --learning_rate 1e-5 \
#         --pad_to_max_length \
#         --beta_schedule squaredcos_improved_ddpm \
#         --weight_decay 0.01 \
#         --top_p 0.99 \
#         --max_steps 50000 \
#         --gradient_accumulation_steps 16 \
#         --warmup_ratio 0.05 \
#         --logging_steps 50 \
#         --save_steps 1000 \
#         --conditional_generation ul2 \
#         --self_condition "logits_mean" \
#         --beaker \
#         --self_condition_mix_before_weights \
#         --dataset_name c4 --streaming --dataset_config_name en


# EXP_NAME="ul2_orig_100k_c4_roberta_base_pretrain_50k_cdcd_fixed"

# gantry run -y -n $EXP_NAME -t $EXP_NAME --allow-dirty \
#     --workspace ai2/tess2 \
#     --nfs \
#     --gpus 1 \
#     --priority high \
#     --cluster ai2/allennlp-cirrascale \
#     --env 'HF_HOME=/net/nfs.cirrascale/allennlp/hamishi/.hf' \
#     --env 'PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python' \
#     --beaker-image 'ai2/pytorch2.0.0-cuda11.8-python3.10' \
#     --venv 'base' \
#     --pip requirements.txt \
#     -- python -m sdlm.run_mlm \
#         --model_name_or_path roberta-base \
#         --per_device_train_batch_size 24  \
#         --per_device_eval_batch_size 24 \
#         --do_train \
#         --do_eval \
#         --output_dir /results \
#         --evaluation_strategy steps \
#         --eval_steps 1000 \
#         --report_to tensorboard \
#         --run_name $EXP_NAME \
#         --max_seq_length 512  \
#         --max_eval_samples 512 \
#         --simplex_value 5 \
#         --num_diffusion_steps 5000  \
#         --num_inference_diffusion_steps 10 100 \
#         --lr_scheduler_type cosine \
#         --learning_rate 1e-5 \
#         --pad_to_max_length \
#         --beta_schedule squaredcos_improved_ddpm \
#         --weight_decay 0.01 \
#         --top_p 0.99 \
#         --max_steps 50000 \
#         --gradient_accumulation_steps 16 \
#         --warmup_ratio 0.05 \
#         --logging_steps 50 \
#         --save_steps 1000 \
#         --conditional_generation ul2 \
#         --self_condition "logits_mean" \
#         --beaker \
#         --self_condition_mix_before_weights \
#         --use_model cdcd \
#         --dataset_name c4 --streaming --dataset_config_name en


# EXP_NAME="ul2_orig_100k_c4_roberta_base_pretrain_50k_tokenwisecdcd_fixed"

# gantry run -y -n $EXP_NAME -t $EXP_NAME --allow-dirty \
#     --workspace ai2/tess2 \
#     --nfs \
#     --gpus 1 \
#     --priority high \
#     --cluster ai2/allennlp-cirrascale \
#     --env 'HF_HOME=/net/nfs.cirrascale/allennlp/hamishi/.hf' \
#     --env 'PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python' \
#     --beaker-image 'ai2/pytorch2.0.0-cuda11.8-python3.10' \
#     --venv 'base' \
#     --pip requirements.txt \
#     -- python -m sdlm.run_mlm \
#         --model_name_or_path roberta-base \
#         --per_device_train_batch_size 24  \
#         --per_device_eval_batch_size 24 \
#         --do_train \
#         --do_eval \
#         --output_dir /results \
#         --evaluation_strategy steps \
#         --eval_steps 1000 \
#         --report_to tensorboard \
#         --run_name $EXP_NAME \
#         --max_seq_length 512  \
#         --max_eval_samples 512 \
#         --simplex_value 5 \
#         --num_diffusion_steps 5000  \
#         --num_inference_diffusion_steps 10 100 \
#         --lr_scheduler_type cosine \
#         --learning_rate 1e-5 \
#         --pad_to_max_length \
#         --beta_schedule squaredcos_improved_ddpm \
#         --weight_decay 0.01 \
#         --top_p 0.99 \
#         --max_steps 50000 \
#         --gradient_accumulation_steps 16 \
#         --warmup_ratio 0.05 \
#         --logging_steps 50 \
#         --save_steps 1000 \
#         --conditional_generation ul2 \
#         --self_condition "logits_mean" \
#         --beaker \
#         --self_condition_mix_before_weights \
#         --use_model tokenwise_cdcd \
#         --dataset_name c4 --streaming --dataset_config_name en


# this was for uncondtional + prefix, but didnt work well.

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

# jake: dolma mistral
gantry run -y -n fineweb_mistral -t fineweb_mistral --allow-dirty \
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
    -- python -m sdlm.run_mlm \
        --model_name_or_path mistralai/Mistral-7B-v0.1 \
        --per_device_train_batch_size 16  \
        --per_device_eval_batch_size 16 \
        --do_train \
        --do_eval \
        --output_dir /results \
        --evaluation_strategy steps \
        --eval_steps 512 \
        --report_to tensorboard \
        --max_seq_length 512  \
        --max_eval_samples 512 \
        --simplex_value 5 \
        --num_diffusion_steps 5000  \
        --num_inference_diffusion_steps 10 100 200 \
        --lr_scheduler_type cosine \
        --learning_rate 1e-5 \
        --pad_to_max_length \
        --beta_schedule squaredcos_improved_ddpm \
        --weight_decay 0.01 \
        --top_p 0.99 \
        --max_steps 100000 \
        --gradient_accumulation_steps 4 \
        --warmup_ratio 0.05 \
        --logging_steps 50 \
        --save_steps 1000 \
        --save_total_limit 3 \
        --conditional_generation ul2 \
        --dataset_name HuggingFaceFW/fineweb --dataset_config_name CC-MAIN-2024-10 --streaming \
        --beaker \
        --bf16 \
        --optim adamw_torch_fused \
        --gradient_checkpointing \
        --use_flash_attention2 \
        --is_causal false \
        --line_by_line true \
        --eval_long_only true \
        --mask_padding_in_loss false \
        --time_embed_lr 1e-4 \
        --freeze_model true


# python -m sdlm.run_mlm \
#         --model_name_or_path mistralai/Mistral-7B-v0.1 \
#         --per_device_train_batch_size 16  \
#         --per_device_eval_batch_size 16 \
#         --do_train \
#         --do_eval \
#         --output_dir outputs/mistral/test \
#         --evaluation_strategy steps \
#         --eval_steps 1 \
#         --report_to tensorboard \
#         --max_seq_length 512  \
#         --max_eval_samples 16 \
#         --simplex_value 5 \
#         --num_diffusion_steps 5000  \
#         --num_inference_diffusion_steps 100 500 \
#         --lr_scheduler_type cosine \
#         --learning_rate 1e-5 \
#         --pad_to_max_length \
#         --beta_schedule squaredcos_improved_ddpm \
#         --weight_decay 0.01 \
#         --top_p 0.99 \
#         --max_steps 100000 \
#         --gradient_accumulation_steps 1 \
#         --warmup_ratio 0.05 \
#         --logging_steps 50 \
#         --save_steps 1000 \
#         --save_total_limit 3 \
#         --conditional_generation ul2 \
#         --dataset_name HuggingFaceFW/fineweb --dataset_config_name CC-MAIN-2024-10 --streaming \
#         --beaker \
#         --bf16 \
#         --optim adamw_torch_fused \
#         --gradient_checkpointing \
#         --use_flash_attention2 \
#         --is_causal false \
#         --line_by_line true \
#         --mask_padding_in_loss false
