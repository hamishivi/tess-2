# Tune temperature.

# Tunes the top-p for the model with length=50.

BASE_DIR="/net/nfs.cirrascale/s2-research/rabeehk/"
params=" --per_device_train_batch_size 24 --per_device_eval_batch_size 25  --do_eval  --evaluation_strategy steps --eval_steps 1000 --report_to tensorboard --overwrite_output_dir  --simplex_value 5 --num_diffusion_steps 5000 --num_inference_diffusion_steps 1000 --lr_scheduler_type linear --learning_rate 1e-4 --pad_to_max_length --beta_schedule squaredcos_improved_ddpm --weight_decay 0.01 --tokenized_data_path processed_data/openwebtext_256_split_gpt_eval/  --max_steps 100000 --gradient_accumulation_steps 8 --warmup_steps 2000 --logging_steps 50 --save_steps 1000 --conditional_generation ul2"
extra_params="--load_states_in_eval_from_model_path --eval_context_size 25 --skip_special_tokens True"
DEBUG_PARAMS="--eval_steps 2 --num_inference_diffusion_steps 3 --per_device_train_batch_size 12 --max_eval_samples 6 --without_compute_metrics --gradient_accumulation_steps 1"
PARAMS_FOR_LOCAL=" --save_total_limit 1 "


# self condition with addition with guidance
TOP_P=0.95 
truncation_length=56
TEMPERATURE=1.0
checkpoint="checkpoint-10000"
model_path="self_condition_with_addition_guidance/"${checkpoint}"/"
for guidance_scale in 2.0 4.0 8.0 10.0 16.0
do
   output_dir=$BASE_DIR"/outputs/paper_experiments/tune_guidance/ul2_self_condition_with_addition_guidance_context_25_top_p_"${TOP_P}"_temperature_"${TEMPERATURE}"_truncation_"${truncation_length}"_guidance_"${guidance_scale}"/"${checkpoint}
   python  -m torch.distributed.launch --nproc_per_node 4 run_mlm.py --model_name_or_path ${model_path} --temperature ${TEMPERATURE} --max_seq_length 256 --truncation_length ${truncation_length} --output_dir ${output_dir} ${params} ${PARAMS_FOR_LOCAL} --eval_context_size 25  ${extra_params}  --top_p ${TOP_P} --self_condition logits_addition  --guidance_scale ${guidance_scale} --max_eval_samples 1000
done

for guidance_scale in 2.0 4.0 8.0 10.0 16.0
do
   output_dir=$BASE_DIR"/outputs/paper_experiments/tune_guidance/ul2_self_condition_with_addition_guidance_context_25_top_p_"${TOP_P}"_temperature_"${TEMPERATURE}"_truncation_"${truncation_length}"_guidance_"${guidance_scale}"/"${checkpoint}
   CUDA_VISIBLE_DEVICES=0 python compute_mlm_metrics.py --model_name_or_path ${model_path} --temperature ${TEMPERATURE} --max_seq_length 256 --truncation_length ${truncation_length}  --output_dir ${output_dir} ${params} ${PARAMS_FOR_LOCAL} --eval_context_size 25  ${extra_params} --top_p ${TOP_P} --self_condition logits_addition --guidance_scale ${guidance_scale} --eval_for_all_metrics --max_eval_samples 1000
done


