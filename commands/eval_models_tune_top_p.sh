# Tunes the top-p for the model with length=50.

BASE_DIR="/net/nfs.cirrascale/s2-research/rabeehk/"
params_for_length_50=" --per_device_train_batch_size 24 --per_device_eval_batch_size 25  --do_eval  --evaluation_strategy steps --eval_steps 1000 --report_to tensorboard --overwrite_output_dir   --simplex_value 5 --num_diffusion_steps 5000 --num_inference_diffusion_steps 1000 --lr_scheduler_type linear --learning_rate 3e-5 --pad_to_max_length --beta_schedule squaredcos_improved_ddpm --weight_decay 0.01  --max_steps 200000 --gradient_accumulation_steps 1 --warmup_steps 2000 --logging_steps 50 --save_steps 1000 --conditional_generation prefix_lm --tokenized_data_path processed_data/openwebtext_256_split_gpt_eval/"
extra_params="--load_states_in_eval_from_model_path --eval_context_size 25 --skip_special_tokens True"
DEBUG_PARAMS="--eval_steps 2 --num_inference_diffusion_steps 3 --per_device_train_batch_size 12 --max_eval_samples 6 --without_compute_metrics --gradient_accumulation_steps 1" 
PARAMS_FOR_LOCAL=" --save_total_limit 1 "
TOP_P=0.99



# DEBUG MODEL trained on length=50 with prefix_lm. 
truncation_length=206
model_path="length_50/checkpoint-102000"
for TOP_P in 0.7 0.9 0.95 0.99  #0.0 0.1 0.2 0.5 0.7 0.9 0.95 0.99
do
    python -m torch.distributed.launch --nproc_per_node 4  run_mlm.py --model_name_or_path ${model_path} --max_seq_length 256 --truncation_length ${truncation_length} --max_eval_samples 1000 --output_dir $BASE_DIR"/outputs/paper_experiments/tune_top_p/ul2_length_50_context_25_generations_"${TOP_P} ${params_for_length_50} ${PARAMS_FOR_LOCAL} --eval_context_size 25  ${extra_params}  --top_p ${TOP_P}
done


for TOP_P in 0.0 0.1 0.2 0.5 0.7 0.9 0.95 0.99
do
    CUDA_VISIBLE_DEVICES=0 python compute_mlm_metrics.py --model_name_or_path ${model_path} --max_seq_length 256 --truncation_length ${truncation_length} --max_eval_samples 1000               --output_dir $BASE_DIR"/outputs/paper_experiments/tune_top_p/ul2_length_50_context_25_generations_"${TOP_P} ${params_for_length_50} ${PARAMS_FOR_LOCAL} --eval_context_size 25  ${extra_params} --eval_for_all_metrics
done

