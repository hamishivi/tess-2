
checkpoint_mount="XXXXXXX:checkpoint-1000:/model"

for task in mnli mrpc qnli qqp rte sst2 stsb
do 
    EXP_NAME="${task}_orig_100k_c4_roberta_base_long_context_short_gen"
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
        -- python -m sdlm.run_glue \
            --model_name_or_path roberta-base \
            --dataset_name mnli \
            --output_dir test \
            --do_train \
            --do_eval \
            --max_seq_length 128 \
            --per_device_train_batch_size 32 \
            --per_device_eval_batch_size 32 \
            --evaluation_strategy epoch \
            --save_strategy steps \
            --report_to tensorboard \
            --overwrite_output_dir \
            --pad_to_max_length \
            --simplex_value 5 \
            --max_train_samples 5000 \
            --num_train_epochs 5 \
            --num_diffusion_steps 5000 \
            --num_inference_diffusion_steps 500 \
            --conditional_generation seq2seq \
            --learning_rate 3e-5 \
            --gradient_accumulation_steps 1 \
            --lr_scheduler_type cosine \
            --beta_schedule squaredcos_improved_ddpm \
            --top_p 0.99 \
            --warmup_ratio 0.03 \
            --logging_steps 50 \
            --save_total_limit 1 \
            --max_eval_samples 500
done

# and for sni, but this is not few-shot.
EXP_NAME="sni_orig_100k_c4_roberta_base_long_context_short_gen"

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
    -- python -m sdlm.run_glue \
        --model_name_or_path roberta-base \
        --dataset_name sni \
        --output_dir /results \
        --do_train \
        --do_eval \
        --max_seq_length 512 \
        --per_device_train_batch_size 32 \
        --per_device_eval_batch_size 32 \
        --evaluation_strategy steps \
        --eval_steps 2000 \
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
        --gradient_accumulation_steps 1 \
        --lr_scheduler_type cosine \
        --beta_schedule squaredcos_improved_ddpm \
        --top_p 0.99 \
        --warmup_ratio 0.03 \
        --logging_steps 50 \
        --save_total_limit 1 \
        --max_eval_samples 1000