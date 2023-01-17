# Tune temperature.

# Tunes the top-p for the model with length=50.

BASE_DIR="/net/nfs.cirrascale/s2-research/rabeehk/"
params=" --per_device_train_batch_size 24 --per_device_eval_batch_size 25  --do_eval  --evaluation_strategy steps --eval_steps 1000 --report_to tensorboard --overwrite_output_dir  --simplex_value 5 --num_diffusion_steps 5000 --num_inference_diffusion_steps 1000 --lr_scheduler_type linear --learning_rate 1e-4 --pad_to_max_length --beta_schedule squaredcos_improved_ddpm --weight_decay 0.01 --tokenized_data_path processed_data/openwebtext_256_split_gpt_eval/  --max_steps 100000 --gradient_accumulation_steps 8 --warmup_steps 2000 --logging_steps 50 --save_steps 1000 --conditional_generation ul2"
extra_params="--load_states_in_eval_from_model_path --eval_context_size 25 --skip_special_tokens True"
DEBUG_PARAMS="--eval_steps 2 --num_inference_diffusion_steps 3 --per_device_train_batch_size 12 --max_eval_samples 6 --without_compute_metrics --gradient_accumulation_steps 1" 
PARAMS_FOR_LOCAL=" --save_total_limit 1 "


# We are considering the models with self-condition addition.
TOP_P=0.95 # 0.95 0.99 # 0.1 0.5 0.7  these are not good
truncation_length=56

: '
checkpoint="checkpoint-15000"
model_path="self_condition_with_addition/"${checkpoint}"/"
for TEMPERATURE in 1.0 2.0 4.0 10.0   #0.1 0.5 1.0 2.0 4.0 10.0 
do
   output_dir=$BASE_DIR"/outputs/paper_experiments/tune_temperatures/ul2_self_condition_addition_context_25_top_p_"${TOP_P}"_temperature_"${TEMPERATURE}"_truncation_"${truncation_length}"/"${checkpoint} 
   # python  -m torch.distributed.launch --nproc_per_node 4 run_mlm.py --model_name_or_path ${model_path} --temperature ${TEMPERATURE} --max_seq_length 256 --truncation_length ${truncation_length} --max_eval_samples 1000 --output_dir ${output_dir} ${params} ${PARAMS_FOR_LOCAL} --eval_context_size 25  ${extra_params}  --top_p ${TOP_P} --self_condition logits_addition  
done


for TEMPERATURE in 1.0 2.0 4.0 10.0 
do
   output_dir=$BASE_DIR"/outputs/paper_experiments/tune_temperatures/ul2_self_condition_addition_context_25_top_p_"${TOP_P}"_temperature_"${TEMPERATURE}"_truncation_"${truncation_length}"/"${checkpoint} 
   CUDA_VISIBLE_DEVICES=0 python compute_mlm_metrics.py --model_name_or_path ${model_path} --temperature ${TEMPERATURE} --max_seq_length 256 --truncation_length ${truncation_length} --max_eval_samples 1000  --output_dir ${output_dir} ${params} ${PARAMS_FOR_LOCAL} --eval_context_size 25  ${extra_params} --top_p ${TOP_P} --self_condition logits_addition --eval_for_all_metrics 
done
'

# Original self-condition. 
: '
checkpoint="checkpoint-15000"
model_path="self_condition/"${checkpoint}"/"
for TEMPERATURE in 1.0 2.0 4.0 10.0  
do
   output_dir=$BASE_DIR"/outputs/paper_experiments/tune_temperatures/ul2_self_condition_context_25_top_p_"${TOP_P}"_temperature_"${TEMPERATURE}"_truncation_"${truncation_length}"/"${checkpoint} 
	# python  -m torch.distributed.launch --nproc_per_node 4 run_mlm.py --model_name_or_path ${model_path} --temperature ${TEMPERATURE} --max_seq_length 256 --truncation_length ${truncation_length} --max_eval_samples 1000 --output_dir ${output_dir} ${params} ${PARAMS_FOR_LOCAL} --eval_context_size 25  ${extra_params}  --top_p ${TOP_P} --self_condition logits
done

for TEMPERATURE in 1.0 2.0 4.0 10.0 
do
   output_dir=$BASE_DIR"/outputs/paper_experiments/tune_temperatures/ul2_self_condition_context_25_top_p_"${TOP_P}"_temperature_"${TEMPERATURE}"_truncation_"${truncation_length}"/"${checkpoint} 
   CUDA_VISIBLE_DEVICES=0 python compute_mlm_metrics.py --model_name_or_path ${model_path} --temperature ${TEMPERATURE} --max_seq_length 256 --truncation_length ${truncation_length} --max_eval_samples 1000  --output_dir ${output_dir} ${params} ${PARAMS_FOR_LOCAL} --eval_context_size 25  ${extra_params} --top_p ${TOP_P} --self_condition logits --eval_for_all_metrics 
done  
'

# ul2.
: '
checkpoint="checkpoint-19000"
model_path="ul2/"${checkpoint}"/"
for TEMPERATURE in 1.0 2.0 4.0 10.0  
do
   output_dir=$BASE_DIR"/outputs/paper_experiments/tune_temperatures/ul2_context_25_top_p_"${TOP_P}"_temperature_"${TEMPERATURE}"_truncation_"${truncation_length}"/"${checkpoint} 
   # python  -m torch.distributed.launch --nproc_per_node 4 run_mlm.py --model_name_or_path ${model_path} --temperature ${TEMPERATURE} --max_seq_length 256 --truncation_length ${truncation_length} --max_eval_samples 1000 --output_dir ${output_dir} ${params} ${PARAMS_FOR_LOCAL} --eval_context_size 25  ${extra_params}  --top_p ${TOP_P} 
done

for TEMPERATURE in 1.0 2.0 4.0 10.0 
do
   output_dir=$BASE_DIR"/outputs/paper_experiments/tune_temperatures/ul2_context_25_top_p_"${TOP_P}"_temperature_"${TEMPERATURE}"_truncation_"${truncation_length}"/"${checkpoint} 
   CUDA_VISIBLE_DEVICES=0 python compute_mlm_metrics.py --model_name_or_path ${model_path} --temperature ${TEMPERATURE} --max_seq_length 256 --truncation_length ${truncation_length} --max_eval_samples 1000  --output_dir ${output_dir} ${params} ${PARAMS_FOR_LOCAL} --eval_context_size 25  ${extra_params} --top_p ${TOP_P}  --eval_for_all_metrics 
done  
'

# self condition with addition with guidance
: '
checkpoint="checkpoint-7000"
model_path="self_condition_with_addition_guidance/"${checkpoint}"/"
for TEMPERATURE in 1.0 2.0 4.0 10.0   #0.1 0.5 1.0 2.0 4.0 10.0 
do
   output_dir=$BASE_DIR"/outputs/paper_experiments/tune_temperatures/ul2_self_condition_with_addition_guidance_2_context_25_top_p_"${TOP_P}"_temperature_"${TEMPERATURE}"_truncation_"${truncation_length}"/"${checkpoint} 
   #python  -m torch.distributed.launch --nproc_per_node 4 run_mlm.py --model_name_or_path ${model_path} --temperature ${TEMPERATURE} --max_seq_length 256 --truncation_length ${truncation_length} --max_eval_samples 1000 --output_dir ${output_dir} ${params} ${PARAMS_FOR_LOCAL} --eval_context_size 25  ${extra_params}  --top_p ${TOP_P} --self_condition logits_addition  --guidance_scale 2
done

for TEMPERATURE in 1.0 2.0 4.0 10.0 
do
   output_dir=$BASE_DIR"/outputs/paper_experiments/tune_temperatures/ul2_self_condition_with_addition_guidance_2_context_25_top_p_"${TOP_P}"_temperature_"${TEMPERATURE}"_truncation_"${truncation_length}"/"${checkpoint} 
   CUDA_VISIBLE_DEVICES=0 python compute_mlm_metrics.py --model_name_or_path ${model_path} --temperature ${TEMPERATURE} --max_seq_length 256 --truncation_length ${truncation_length} --max_eval_samples 1000  --output_dir ${output_dir} ${params} ${PARAMS_FOR_LOCAL} --eval_context_size 25  ${extra_params} --top_p ${TOP_P} --self_condition logits_addition --guidance_scale 2 --eval_for_all_metrics 
done
'

#############################################
# Running for 5K examples.
#############################################
TOP_P=0.95
truncation_length=56
TEMPERATURE=1.0

: '
checkpoint="checkpoint-22000"
model_path="self_condition_with_addition/"${checkpoint}"/"
output_dir=$BASE_DIR"/outputs/paper_experiments/5k_eval/ul2_self_condition_addition_context_25_top_p_"${TOP_P}"_temperature_"${TEMPERATURE}"_truncation_"${truncation_length}"/"${checkpoint} 
python  -m torch.distributed.launch --nproc_per_node 4 run_mlm.py --model_name_or_path ${model_path} --temperature ${TEMPERATURE} --max_seq_length 256 --truncation_length ${truncation_length}  --output_dir ${output_dir} ${params} ${PARAMS_FOR_LOCAL} --eval_context_size 25  ${extra_params}  --top_p ${TOP_P} --self_condition logits_addition --max_eval_samples 5000 
CUDA_VISIBLE_DEVICES=0 python compute_mlm_metrics.py --model_name_or_path ${model_path} --temperature ${TEMPERATURE} --max_seq_length 256 --truncation_length ${truncation_length}   --output_dir ${output_dir} ${params} ${PARAMS_FOR_LOCAL} --eval_context_size 25  ${extra_params} --top_p ${TOP_P} --self_condition logits_addition --eval_for_all_metrics --max_eval_samples 5000 
'

: '
# Original self-condition. 
checkpoint="checkpoint-23000"
model_path="self_condition/"${checkpoint}"/"
output_dir=$BASE_DIR"/outputs/paper_experiments/5k_eval/ul2_self_condition_context_25_top_p_"${TOP_P}"_temperature_"${TEMPERATURE}"_truncation_"${truncation_length}"/"${checkpoint} 
python  -m torch.distributed.launch --nproc_per_node 4 run_mlm.py --model_name_or_path ${model_path} --temperature ${TEMPERATURE} --max_seq_length 256 --truncation_length ${truncation_length}  --output_dir ${output_dir} ${params} ${PARAMS_FOR_LOCAL} --eval_context_size 25  ${extra_params}  --top_p ${TOP_P} --self_condition logits --max_eval_samples 5000 
CUDA_VISIBLE_DEVICES=0 python compute_mlm_metrics.py --model_name_or_path ${model_path} --temperature ${TEMPERATURE} --max_seq_length 256 --truncation_length ${truncation_length}  --output_dir ${output_dir} ${params} ${PARAMS_FOR_LOCAL} --eval_context_size 25  ${extra_params} --top_p ${TOP_P} --self_condition logits --eval_for_all_metrics  --max_eval_samples 5000 
'

# ul2.
checkpoint="checkpoint-30000"
model_path="ul2/"${checkpoint}"/"
output_dir=$BASE_DIR"/outputs/paper_experiments/5k_eval/ul2_context_25_top_p_"${TOP_P}"_temperature_"${TEMPERATURE}"_truncation_"${truncation_length}"/"${checkpoint} 
python  -m torch.distributed.launch --nproc_per_node 4 run_mlm.py --model_name_or_path ${model_path} --temperature ${TEMPERATURE} --max_seq_length 256 --truncation_length ${truncation_length}  --output_dir ${output_dir} ${params} ${PARAMS_FOR_LOCAL} --eval_context_size 25  ${extra_params}  --top_p ${TOP_P}  --max_eval_samples 5000
CUDA_VISIBLE_DEVICES=0 python compute_mlm_metrics.py --model_name_or_path ${model_path} --temperature ${TEMPERATURE} --max_seq_length 256 --truncation_length ${truncation_length} --output_dir ${output_dir} ${params} ${PARAMS_FOR_LOCAL} --eval_context_size 25  ${extra_params} --top_p ${TOP_P}  --eval_for_all_metrics  --max_eval_samples 5000
