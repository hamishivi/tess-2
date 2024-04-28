# experimenting over sni
# EXP_NAME="sni_rbase"
# checkpoint_mount="01H4KVBDMMN284JQ6G2N6GS2EV:checkpoint-10000"

gantry run -y -n tess_self_cond_tulu2_test -t tess_self_cond_tulu2_test --allow-dirty \
    --workspace ai2/tess2 \
    --nfs \
    --gpus 1 \
    --budget ai2/allennlp \
    --priority normal \
    --cluster ai2/allennlp-cirrascale \
    --env 'HF_HOME=/net/nfs.cirrascale/allennlp/hamishi/.hf' \
    --env 'PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python' \
    --env 'IS_ALPACA_EVAL_2=False' \
    --beaker-image 'ai2/pytorch2.0.0-cuda11.8-python3.10' \
    --venv 'base' \
    --pip requirements.txt \
    -- python -m sdlm.run_instruction_tuning_tulu \
    --model_name_or_path roberta-base \
    --dataset_name allenai/tulu-v2-sft-mixture \
    --output_dir /results \
    --do_train \
    --do_eval \
    --max_seq_length 512 \
    --per_device_train_batch_size 16 \
    --skip_special_tokens False \
    --per_device_eval_batch_size 16 \
    --evaluation_strategy epoch \
    --save_strategy steps \
    --report_to tensorboard \
    --overwrite_output_dir \
    --pad_to_max_length \
    --simplex_value 5 \
    --num_train_epochs 2 \
    --num_diffusion_steps 5000 \
    --num_inference_diffusion_steps 100 \
    --conditional_generation seq2seq \
    --learning_rate 3e-5 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --beta_schedule squaredcos_improved_ddpm \
    --top_p 0.99 \
    --warmup_ratio 0.03 \
    --logging_steps 50 \
    --save_total_limit 1 \
    --max_eval_samples 1000 \
    --self_condition "logits_mean" \
    --self_condition_mix_before_weights
