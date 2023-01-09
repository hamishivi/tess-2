# Tune temperature.

# Tunes the top-p for the model with length=50.

BASE_DIR="/net/nfs.cirrascale/s2-research/rabeehk/"
params=" --per_device_train_batch_size 24 --per_device_eval_batch_size 25  --do_eval  --evaluation_strategy steps --eval_steps 1000 --report_to tensorboard --overwrite_output_dir  --simplex_value 5 --num_diffusion_steps 5000 --num_inference_diffusion_steps 1000 --lr_scheduler_type linear --learning_rate 1e-4 --pad_to_max_length --beta_schedule squaredcos_improved_ddpm --weight_decay 0.01 --tokenized_data_path processed_data/openwebtext_256_split_gpt_eval/  --max_steps 100000 --gradient_accumulation_steps 8 --warmup_steps 2000 --logging_steps 50 --save_steps 1000 --conditional_generation ul2"
extra_params="--load_states_in_eval_from_model_path --eval_context_size 25 --skip_special_tokens True"
DEBUG_PARAMS="--eval_steps 2 --num_inference_diffusion_steps 3 --per_device_train_batch_size 12 --max_eval_samples 6 --without_compute_metrics --gradient_accumulation_steps 1" 
PARAMS_FOR_LOCAL=" --save_total_limit 1 "


# We are considering the models with self-condition addition.
TOP_P=0.9 # 0.1 0.5 0.7 0.9
truncation_length=0
model_path="self_condition_with_addition/checkpoint-8000/"
for TEMPERATURE in 0.1 0.5 1.0 2.0 4.0 10.0 
do
	python  -m torch.distributed.launch --nproc_per_node 4 run_mlm.py --model_name_or_path ${model_path} --temperature ${TEMPERATURE} --max_seq_length 256 --truncation_length ${truncation_length} --max_eval_samples 1000 --output_dir $BASE_DIR"/outputs/paper_experiments/tune_temperature/ul2_self_condition_addition_context_25_generations_"${TOP_P}"_temperature_"${TEMPERATURE} ${params} ${PARAMS_FOR_LOCAL} --eval_context_size 25  ${extra_params}  --top_p ${TOP_P} --self_condition logits_addition  
done


for TEMPERATURE in 0.1 0.5 1.0 2.0 4.0 10.0 
do
   CUDA_VISIBLE_DEVICES=0 python compute_mlm_metrics.py --model_name_or_path ${model_path} --temperature ${TEMPERATURE} --max_seq_length 256 --truncation_length ${truncation_length} --max_eval_samples 1000  --output_dir $BASE_DIR"/outputs/paper_experiments/tune_temperature/ul2_self_condition_addition_context_25_generations_"${TOP_P}"_temperature_"${TEMPERATURE} ${params} ${PARAMS_FOR_LOCAL} --eval_context_size 25  ${extra_params} --top_p ${TOP_P} --self_condition logits_addition --eval_for_all_metrics 
done   


