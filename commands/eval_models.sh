
BASE_DIR="/net/nfs.cirrascale/s2-research/rabeehk/"
shared_params=" --per_device_train_batch_size 24 --per_device_eval_batch_size 6  --do_eval  --evaluation_strategy steps --eval_steps 1000 --report_to tensorboard --overwrite_output_dir  --max_eval_samples 48 --simplex_value 5 --num_diffusion_steps 5000 --num_inference_diffusion_steps 1000 --lr_scheduler_type linear --learning_rate 1e-4 --pad_to_max_length --beta_schedule squaredcos_improved_ddpm --weight_decay 0.01 --tokenized_data_path processed_data/openwebtext_256_split_gpt_eval/ --top_p 0.99 --max_steps 100000 --gradient_accumulation_steps 8 --warmup_steps 2000 --logging_steps 50 --save_steps 1000 --conditional_generation ul2"


params_for_length_50=" --per_device_train_batch_size 24 --per_device_eval_batch_size 25  --do_eval  --evaluation_strategy steps --eval_steps 1000 --report_to tensorboard --overwrite_output_dir   --simplex_value 5 --num_diffusion_steps 5000 --num_inference_diffusion_steps 1000 --lr_scheduler_type linear --learning_rate 3e-5 --pad_to_max_length --beta_schedule squaredcos_improved_ddpm --weight_decay 0.01 --tokenized_data_path processed_data/openwebtext_50_split/ --top_p 0.99 --max_steps 200000 --gradient_accumulation_steps 1 --warmup_steps 2000 --logging_steps 50 --save_steps 1000 --conditional_generation prefix_lm"
extra_params="--load_states_in_eval_from_model_path --eval_context_size 25 --skip_special_tokens True"
params_for_simple_data=" --per_device_train_batch_size 24 --per_device_eval_batch_size 6  --do_eval  --evaluation_strategy steps --eval_steps 1000 --report_to tensorboard --overwrite_output_dir  --max_eval_samples 48 --simplex_value 5 --num_diffusion_steps 5000 --num_inference_diffusion_steps 1000 --lr_scheduler_type linear --learning_rate 1e-4 --pad_to_max_length --beta_schedule squaredcos_improved_ddpm --weight_decay 0.01  --top_p 0.99 --max_steps 60000 --gradient_accumulation_steps 1 --warmup_steps 2000 --logging_steps 50 --save_steps 1000     --train_file /net/nfs.cirrascale/s2-research/rabeehk/diffusion/small_data/simple-train.txt --validation_file /net/nfs.cirrascale/s2-research/rabeehk/diffusion/small_data/simple-test.txt"

DEBUG_PARAMS="--eval_steps 2 --num_inference_diffusion_steps 3 --per_device_train_batch_size 12 --max_eval_samples 6 --without_compute_metrics --gradient_accumulation_steps 1" 
PARAMS_FOR_LOCAL=" --save_total_limit 1 "
truncation_length=0
TOP_P=0.99





# DEBUG MODEL trained on length=50 with prefix_lm. 
#truncation_length=0
#model_path="checkpoint-57000"
#python -m torch.distributed.launch --nproc_per_node 4  run_mlm.py --model_name_or_path ${model_path} --max_seq_length 50 --truncation_length ${truncation_length} --max_eval_samples 100 --output_dir $BASE_DIR"/outputs/paper_experiments/ul2_length_50_context_25_generations_"${TOP_P} ${params_for_length_50} ${PARAMS_FOR_LOCAL} --eval_context_size 25  ${extra_params} 
#CUDA_VISIBLE_DEVICES=0 python compute_mlm_metrics.py --model_name_or_path ${model_path} --max_seq_length 50 --truncation_length ${truncation_length} --max_eval_samples 100 --output_dir $BASE_DIR"/outputs/paper_experiments/ul2_length_50_context_25_generations_"${TOP_P} ${params_for_length_50} ${PARAMS_FOR_LOCAL} --eval_context_size 25  ${extra_params} --eval_for_all_metrics



truncation_length=56
TOP_P=0.95
shared_params=" --per_device_train_batch_size 24 --per_device_eval_batch_size 25  --do_eval  --evaluation_strategy steps --eval_steps 1000 --report_to tensorboard --overwrite_output_dir --max_seq_length 256 --max_eval_samples 1000 --simplex_value 5 --num_diffusion_steps 5000 --num_inference_diffusion_steps 1000 --lr_scheduler_type linear --top_p 0.99 --learning_rate 1e-4 --pad_to_max_length --beta_schedule squaredcos_improved_ddpm --weight_decay 0.01 --tokenized_data_path processed_data/openwebtext_256_split_gpt_eval/  --max_steps 100000 --gradient_accumulation_steps 8 --warmup_steps 2000 --logging_steps 50 --save_steps 1000 --conditional_generation ul2"

# UL2 model.
checkpoint="checkpoint-15000"
model_path="ul2/"${checkpoint}
TOP_P=0.95
#python -m torch.distributed.launch --nproc_per_node 4 run_mlm.py --model_name_or_path ${model_path} --truncation_length ${truncation_length} --output_dir $BASE_DIR"/outputs/paper_experiments/ul2_"${truncation_length}"_top_p_"${TOP_P}"_context_25_"${model_path} ${shared_params} ${PARAMS_FOR_LOCAL} ${extra_params} --top_p ${TOP_P}
#CUDA_VISIBLE_DEVICES=0 python compute_mlm_metrics.py --model_name_or_path ${model_path} --truncation_length ${truncation_length} --output_dir $BASE_DIR"/outputs/paper_experiments/ul2_"${truncation_length}"_top_p_"${TOP_P}"_context_25_"${model_path}  ${shared_params} ${PARAMS_FOR_LOCAL} ${extra_params} --eval_for_all_metrics --top_p ${TOP_P}

# ul2_with_self_condition
model_path="self_condition/checkpoint-12000/"
TOP_P=0.95
#python -m torch.distributed.launch --nproc_per_node 4 run_mlm.py --model_name_or_path ${model_path} --truncation_length ${truncation_length} --output_dir $BASE_DIR"/outputs/paper_experiments/ul2_self_condition_"${truncation_length}"_top_p_"${TOP_P}"_context_25_"${model_path} ${shared_params} ${PARAMS_FOR_LOCAL} ${extra_params} --self_condition logits 
# python compute_mlm_metrics.py --model_name_or_path ${model_path} --truncation_length ${truncation_length} --output_dir $BASE_DIR"/outputs/paper_experiments/ul2_self_condition_"${truncation_length}"_top_p_"${TOP_P}"_context_25_"${model_path} ${shared_params} ${PARAMS_FOR_LOCAL} ${extra_params} --self_condition logits --eval_for_all_metrics 

# ul2_with_self_condition_with_addition 
TOP_P=0.95
model_path="self_condition_with_addition/checkpoint-11000/"
# python -m torch.distributed.launch --nproc_per_node 4 run_mlm.py --model_name_or_path ${model_path} --truncation_length ${truncation_length} --output_dir $BASE_DIR"/outputs/paper_experiments/ul2_self_condition_with_addition_"${truncation_length}"_top_p_"${TOP_P}"_context_25_"${model_path} ${shared_params} ${PARAMS_FOR_LOCAL} ${extra_params} --self_condition logits_addition  
#CUDA_VISIBLE_DEVICES=0 python compute_mlm_metrics.py --model_name_or_path ${model_path} --truncation_length ${truncation_length} --output_dir $BASE_DIR"/outputs/paper_experiments/ul2_self_condition_with_addition_"${truncation_length}"_top_p_"${TOP_P}"_context_25_"${model_path} ${shared_params} ${PARAMS_FOR_LOCAL} ${extra_params} --self_condition logits_addition  --eval_for_all_metrics 

##########################################################
# Running for a different checkpoint.
truncation_length=56
TOP_P=0.95
model_path="self_condition_with_addition/checkpoint-7000/"
python -m torch.distributed.launch --nproc_per_node 4 run_mlm.py --model_name_or_path ${model_path} --truncation_length ${truncation_length} --output_dir $BASE_DIR"/outputs/paper_experiments/ul2_self_condition_with_addition_"${truncation_length}"_top_p_"${TOP_P}"_context_25_"${model_path} ${shared_params} ${PARAMS_FOR_LOCAL} ${extra_params} --self_condition logits_addition --max_eval_samples 1000 
CUDA_VISIBLE_DEVICES=0 python compute_mlm_metrics.py --model_name_or_path ${model_path} --truncation_length ${truncation_length} --output_dir $BASE_DIR"/outputs/paper_experiments/ul2_self_condition_with_addition_"${truncation_length}"_top_p_"${TOP_P}"_context_25_"${model_path} ${shared_params} ${PARAMS_FOR_LOCAL} ${extra_params} --self_condition logits_addition  --eval_for_all_metrics --max_eval_samples 1000

TOP_P=0.95
model_path="self_condition_with_addition/checkpoint-28000/"
python -m torch.distributed.launch --nproc_per_node 4 run_mlm.py --model_name_or_path ${model_path} --truncation_length ${truncation_length} --output_dir $BASE_DIR"/outputs/paper_experiments/ul2_self_condition_with_addition_"${truncation_length}"_top_p_"${TOP_P}"_context_25_"${model_path} ${shared_params} ${PARAMS_FOR_LOCAL} ${extra_params} --self_condition logits_addition --max_eval_samples 1000 
CUDA_VISIBLE_DEVICES=0 python compute_mlm_metrics.py --model_name_or_path ${model_path} --truncation_length ${truncation_length} --output_dir $BASE_DIR"/outputs/paper_experiments/ul2_self_condition_with_addition_"${truncation_length}"_top_p_"${TOP_P}"_context_25_"${model_path} ${shared_params} ${PARAMS_FOR_LOCAL} ${extra_params} --self_condition logits_addition  --eval_for_all_metrics --max_eval_samples 1000
##########################################################



# UL2 with self-condition with addition with guidance.
model_path="self_condition_with_addition_guidance/checkpoint-7000/"
# python -m torch.distributed.launch --nproc_per_node 4 run_mlm.py --model_name_or_path ${model_path} --truncation_length ${truncation_length} --output_dir $BASE_DIR"/outputs/paper_experiments/ul2_self_condition_with_addition_guidance"${truncation_length}"_top_p_"${TOP_P}"_context_25_"${model_path} ${shared_params} ${PARAMS_FOR_LOCAL} ${extra_params} --self_condition logits_addition  --guidance_scale 2
#python compute_mlm_metrics.py --model_name_or_path ${model_path} --truncation_length ${truncation_length} --output_dir $BASE_DIR"/outputs/paper_experiments/ul2_self_condition_with_addition_guidance"${truncation_length}"_top_p_"${TOP_P}"_context_25_"${model_path} ${shared_params} ${PARAMS_FOR_LOCAL} ${extra_params} --self_condition logits_addition  --guidance_scale 2 --eval_for_all_metrics


: '
# Evaluate the model of self-condition addition on the length=25 on two different checkpoints.
truncation_length=206
for TOP_P in 0.95 0.99 # 0.8 0.9 # 0.95 0.99 # 0.8 0.9 #0.95 0.99 # 0.8 0.9 
do
   for TEMPERATURE in 1 2 4
   do
      checkpoint="/checkpoint-28000/"
      model_path="self_condition_with_addition/"${checkpoint}
      output_dir=$BASE_DIR"/outputs/paper_experiments/tune_length_25_context_25_truncation_"${truncation_length}"/ul2_self_condition_with_addition_top_p_"${TOP_P}"_temperature_"${TEMPERATURE}""${checkpoint}
      python -m torch.distributed.launch --nproc_per_node 4 run_mlm.py --model_name_or_path ${model_path} --truncation_length ${truncation_length} --output_dir ${output_dir} ${shared_params} ${PARAMS_FOR_LOCAL} ${extra_params} --self_condition logits_addition --max_seq_length 256 --truncation_length 206 --max_eval_samples 1000 --per_device_eval_batch_size 25 --temperature ${TEMPERATURE}
   done
done 

for TOP_P in 0.95 0.99 # 0.8 0.9 # 0.95 0.99 #  #  # 0.8 0.9
do
   for TEMPERATURE in 1 2 4
   do
      output_dir=$BASE_DIR"/outputs/paper_experiments/tune_length_25_context_25_truncation_"${truncation_length}"/ul2_self_condition_with_addition_top_p_"${TOP_P}"_temperature_"${TEMPERATURE}""${checkpoint}
      CUDA_VISIBLE_DEVICES=0 python compute_mlm_metrics.py --model_name_or_path ${model_path} --truncation_length ${truncation_length} --output_dir ${output_dir} ${shared_params} ${PARAMS_FOR_LOCAL} ${extra_params} --self_condition logits_addition  --eval_for_all_metrics --max_seq_length 256 --truncation_length 206 --max_eval_samples 1000 --per_device_eval_batch_size 25 --temperature ${TEMPERATURE}
   done
done
'
