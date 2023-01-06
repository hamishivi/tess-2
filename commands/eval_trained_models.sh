# Evaluation of our models trained on the cloud from a checkpoint

TOP_P=0.99
BASE_DIR="/net/nfs.cirrascale/s2-research/"
tokenized_data_path=${BASE_DIR}"rabeehk/simplex-diffusion/processed_data/openwebtext_256_split_gpt_eval/"
shared_params="--without_compute_metrics --per_device_train_batch_size 12 --per_device_eval_batch_size 25  --do_eval  --evaluation_strategy steps --eval_steps 1000 --report_to tensorboard --overwrite_output_dir --max_seq_length 256  --simplex_value 5 --num_diffusion_steps 5000 --num_inference_diffusion_steps 1000 --lr_scheduler_type cosine --learning_rate 1e-4 --pad_to_max_length --beta_schedule squaredcos_improved_ddpm --weight_decay 0.01  --max_steps 2000000 --gradient_accumulation_steps 8 --warmup_steps 2000 --logging_steps 50 --save_steps 1000 --conditional_generation ul2   --eval_for_all_metrics  --load_states_in_eval_from_model_path --eval_context_size 32 --skip_special_tokens True"
output_dir=${BASE_DIR}"rabeehk/outputs/paper_experiments/ours_eval/"
truncation_length=56

# On length = 200
# Eval only on one GPU due to speed issue with MAUVE metric.
# our self-condition addition with guidance eval 
MODEL_PATH="/net/nfs.cirrascale/s2-research/rabeehk/outputs/test/checkpoint-10" 
MODEL_NAME="ul2_self_condition_addition_guidance_5"
python   -m torch.distributed.launch --nproc_per_node 4  run_mlm.py  --truncation_length ${truncation_length} --model_name_or_path ${MODEL_PATH}   --output_dir ${output_dir}/${MODEL_NAME}"_"${TOP_P}"_"${truncation_length}  ${shared_params} --tokenized_data_path ${tokenized_data_path} --top_p ${TOP_P} --self_condition logits_addition --guidance_scale 5 --max_eval_samples 25 --num_inference_diffusion_steps 10 
CUDA_VISIBLE_DEVICES=0 python compute_mlm_metrics.py  --truncation_length ${truncation_length} --model_name_or_path ${MODEL_PATH}   --output_dir ${output_dir}/${MODEL_NAME}"_"${TOP_P}"_"${truncation_length}  ${shared_params} --tokenized_data_path ${tokenized_data_path} --top_p ${TOP_P} --self_condition logits_addition --guidance_scale 5  --max_eval_samples 25 --num_inference_diffusion_steps 10 


# our self-condition addition 
MODEL_NAME="self-condition-addition"
# python -m torch.distributed.launch --nproc_per_node 4 run_mlm.py --truncation_length ${truncation_length} --model_name_or_path ${MODEL_PATH}  --output_dir ${output_dir}/${MODEL_NAME}"_"${TOP_P}"_"${truncation_length} ${shared_params} --tokenized_data_path ${tokenized_data_path} --top_p ${TOP_P} --self_condition logits_addition  
# CUDA_VISIBLE_DEVICES=0 python compute_mlm_metrics.py --truncation_length ${truncation_length} --model_name_or_path ${MODEL_PATH}  --output_dir ${output_dir}/${MODEL_NAME}"_"${TOP_P}"_"${truncation_length} ${shared_params} --tokenized_data_path ${tokenized_data_path} --top_p ${TOP_P} --self_condition logits_addition  

# ul2 model
MODEL_NAME="ul2"
# python -m torch.distributed.launch --nproc_per_node 4 run_mlm.py --truncation_length ${truncation_length} --model_name_or_path ${MODEL_PATH}  --output_dir ${output_dir}/${MODEL_NAME}"_"${TOP_P}"_"${truncation_length}  ${shared_params} --tokenized_data_path ${tokenized_data_path} --top_p ${TOP_P}
# CUDA_VISIBLE_DEVICES=0 python compute_mlm_metrics.py --truncation_length ${truncation_length} --model_name_or_path ${MODEL_PATH}  --output_dir ${output_dir}/${MODEL_NAME}"_"${TOP_P}"_"${truncation_length}  ${shared_params} --tokenized_data_path ${tokenized_data_path} --top_p ${TOP_P}

# original self-condition
MODEL_NAME="self-condition-original"
# python -m torch.distributed.launch --nproc_per_node 4 run_mlm.py --truncation_length ${truncation_length} --model_name_or_path ${MODEL_PATH}  --output_dir ${output_dir}/${MODEL_NAME}"_"${TOP_P}"_"${truncation_length}  ${shared_params} --tokenized_data_path ${tokenized_data_path} --top_p ${TOP_P}  --self_condition logits 
# CUDA_VISIBLE_DEVICES=0 python compute_mlm_metrics.py --truncation_length ${truncation_length} --model_name_or_path ${MODEL_PATH}  --output_dir ${output_dir}/${MODEL_NAME}"_"${TOP_P}"_"${truncation_length} ${shared_params} --tokenized_data_path ${tokenized_data_path} --top_p ${TOP_P}  --self_condition logits 
