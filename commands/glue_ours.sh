
# GLUE should be run with 128+label_length, where label_length=5.
shared_params="--model_name_or_path roberta-large  --do_train --do_eval --do_predict --max_seq_length 133 --per_device_train_batch_size 64 --per_device_eval_batch_size 64 --evaluation_strategy steps --save_strategy steps   --report_to tensorboard --overwrite_output_dir --pad_to_max_length  --simplex_value 5 --num_diffusion_steps 5000 --conditional_generation seq2seq  --learning_rate 3e-5 --gradient_accumulation_steps 2 --lr_scheduler_type linear --beta_schedule squaredcos_improved_ddpm  --top_p 0.99 --warmup_steps 500 --logging_steps 50 --save_steps 1000  --add_t5_tags --max_steps 100000   --load_best_model_at_end true --checkpoint_best_model  --greater_is_better true --eval_steps 1000"
BASE_DIR="/net/nfs.cirrascale/s2-research/rabeehk/"
PARAMS_FOR_LOCAL=" --save_total_limit 1"
DEBUG_PARAMS="--eval_steps 2 --num_inference_diffusion_steps 3 --per_device_train_batch_size 12 --max_eval_samples 6"
num_inference_diffusion_steps=10


# Test weight decay and iterations => with WD was the best with iterations=10
#DATASET="mrpc"
#python run_glue.py  --dataset_name ${DATASET} ${shared_params} --output_dir $BASE_DIR"outputs/paper_experiments/ours_glue/"${DATASET}"_steps_"${num_inference_diffusion_steps}  --num_inference_diffusion_steps ${num_inference_diffusion_steps} ${PARAMS_FOR_LOCAL} 

: '
# Runing this one for all datasets.
# with weight_decay
DATASET="cola"
python run_glue.py  --dataset_name ${DATASET} ${shared_params} --output_dir $BASE_DIR"outputs/paper_experiments/ours_glue/"${DATASET}"_steps_"${num_inference_diffusion_steps}"_wd_0.01"  --num_inference_diffusion_steps ${num_inference_diffusion_steps} ${PARAMS_FOR_LOCAL} --weight_decay 0.01
'

# DEBUG
#DATASET="cola"
#python run_glue.py  --dataset_name ${DATASET} ${shared_params} --output_dir $BASE_DIR"outputs/paper_experiments/ours_glue/debug"  --num_inference_diffusion_steps ${num_inference_diffusion_steps} ${PARAMS_FOR_LOCAL} ${DEBUG_PARAMS} --self_condition_mix_before_weights true --self_condition "logits_multiply" --save_steps 10 --eval_steps 10 

# Training GLUE with self-conditioning mean for now.
: '
DATASET="cola"
python run_glue.py  --dataset_name ${DATASET} ${shared_params} --output_dir $BASE_DIR"outputs/paper_experiments/ours_glue_self_condition_mean/"${DATASET}"_steps_"${num_inference_diffusion_steps}"_wd_0.01"  --num_inference_diffusion_steps ${num_inference_diffusion_steps} ${PARAMS_FOR_LOCAL} --weight_decay 0.01 --self_condition "logits_mean" --per_device_train_batch_size 32  --gradient_accumulation_steps 4
'

# Training GLUE with self-conditioning max for now.
: '
DATASET="qnli"
python -m torch.distributed.launch --nproc_per_node 2 run_glue.py  --dataset_name ${DATASET} ${shared_params} --output_dir $BASE_DIR"outputs/paper_experiments/ours_glue_self_condition_max/"${DATASET}"_steps_"${num_inference_diffusion_steps}"_wd_0.01"  --num_inference_diffusion_steps ${num_inference_diffusion_steps} ${PARAMS_FOR_LOCAL} --weight_decay 0.01 --self_condition "logits_max" --per_device_train_batch_size 32  --gradient_accumulation_steps 2
'

# Training GLUE with self-conditioning max for now.
: '
DATASET="qnli"
python -m torch.distributed.launch --nproc_per_node 2 run_glue.py  --dataset_name ${DATASET} ${shared_params} --output_dir $BASE_DIR"outputs/paper_experiments/ours_glue_self_condition_addition/"${DATASET}"_steps_"${num_inference_diffusion_steps}"_wd_0.01"  --num_inference_diffusion_steps ${num_inference_diffusion_steps} ${PARAMS_FOR_LOCAL} --weight_decay 0.01 --self_condition "logits_addition" --per_device_train_batch_size 32  --gradient_accumulation_steps 2
'

# Running from a checkpoint with self-condition mean.
: '
DATASET="rte"
python run_glue.py  --dataset_name ${DATASET} ${shared_params} --output_dir $BASE_DIR"outputs/paper_experiments/ours_glue_self_condition_mean/"${DATASET}"_steps_"${num_inference_diffusion_steps}"_wd_0.01_from_40K_checkpoint"  --num_inference_diffusion_steps ${num_inference_diffusion_steps} ${PARAMS_FOR_LOCAL} --weight_decay 0.01 --self_condition "logits_mean" --per_device_train_batch_size 32  --gradient_accumulation_steps 4 --model_name_or_path  "self_condition_mean/checkpoint-40000/"
'

##############################################################################
# Running GLUE with the self-condition mean with the mix before weights setup
##############################################################################
# DATASET="qqp"
# num_inference_diffusion_steps=10
#python run_glue.py  --dataset_name ${DATASET} ${shared_params} --output_dir $BASE_DIR"outputs/paper_experiments/glue_results/ours_self_condition_mean_mix_before_weights_"${DATASET}"_steps_"${num_inference_diffusion_steps}"_wd_0.01"  --num_inference_diffusion_steps ${num_inference_diffusion_steps} ${PARAMS_FOR_LOCAL} --weight_decay 0.01 --self_condition "logits_mean"  --per_device_train_batch_size 32  --gradient_accumulation_steps 4  --self_condition_mix_before_weights true --resume_from_checkpoint "/net/nfs.cirrascale/s2-research/rabeehk/outputs/paper_experiments/glue_results/ours_self_condition_mean_mix_before_weights_mnli_steps_10_wd_0.01/checkpoint-12000/"

# NOTE: to run for 4 GPU, modify and remove 4 GPUs. Also remove the copy at the end of the output dir.
# python run_glue.py  --dataset_name ${DATASET} ${shared_params} --output_dir $BASE_DIR"outputs/paper_experiments/glue_results/ours_self_condition_mean_mix_before_weights_"${DATASET}"_steps_"${num_inference_diffusion_steps}"_no_wd"  --num_inference_diffusion_steps ${num_inference_diffusion_steps} ${PARAMS_FOR_LOCAL} --weight_decay 0.0 --self_condition "logits_mean"  --per_device_train_batch_size 32  --gradient_accumulation_steps 4  --self_condition_mix_before_weights true 

# Training without wd for small data with checkpoint of 500 steps.
# NOTE: runs on 2 GPUS, laters modify the grad_acc and then remove the 2 GPUS.
#python -m torch.distributed.launch --nproc_per_node 4 run_glue.py  --dataset_name ${DATASET} ${shared_params} --output_dir $BASE_DIR"outputs/paper_experiments/glue_results/ours_self_condition_mean_mix_before_weights_"${DATASET}"_steps_"${num_inference_diffusion_steps}"_no_wd_500_steps"  --num_inference_diffusion_steps ${num_inference_diffusion_steps} ${PARAMS_FOR_LOCAL} --weight_decay 0.0 --self_condition "logits_mean"  --per_device_train_batch_size 32  --gradient_accumulation_steps 1  --self_condition_mix_before_weights true --save_steps 500 --eval_steps 500


##############################################################################################################################
# Running GLUE with the self-condition mean with the mix before weights setup for the selected max iterations.
##############################################################################################################################

# NOTE: to run for 4 GPU, modify and remove 4 GPUs. Also remove the copy at the end of the output dir.
# For larger datasets 
# sst2, mnli, qnli, qqp
# DATASET="qqp"
# num_inference_diffusion_steps=10
# python -m torch.distributed.launch --nproc_per_node 4 run_glue.py  --dataset_name ${DATASET} ${shared_params} --output_dir $BASE_DIR"outputs/paper_experiments/glue_results/ours_self_condition_mean_mix_before_weights_"${DATASET}"_steps_"${num_inference_diffusion_steps}"_no_wd_max_steps_set"  --num_inference_diffusion_steps ${num_inference_diffusion_steps} ${PARAMS_FOR_LOCAL} --weight_decay 0.0 --self_condition "logits_mean"  --per_device_train_batch_size 32  --gradient_accumulation_steps 1  --self_condition_mix_before_weights true  --max_steps 25000 --save_checkpoints_on_s3 --resume_from_checkpoint "/net/nfs.cirrascale/s2-research/rabeehk/outputs/paper_experiments/glue_results/ours_self_condition_mean_mix_before_weights_qqp_steps_10_no_wd_max_steps_set/checkpoint-23000" 

# For smaller datasets.
# DATASET="cola" # rte, mrpc, cola, stsb, wnli
# python -m torch.distributed.launch --nproc_per_node 4 run_glue.py  --dataset_name ${DATASET} ${shared_params} --output_dir $BASE_DIR"outputs/paper_experiments/glue_results/ours_self_condition_mean_mix_before_weights_"${DATASET}"_steps_"${num_inference_diffusion_steps}"_no_wd_max_steps_set"  --num_inference_diffusion_steps ${num_inference_diffusion_steps} ${PARAMS_FOR_LOCAL} --weight_decay 0.0 --self_condition "logits_mean"  --per_device_train_batch_size 32  --gradient_accumulation_steps 1  --self_condition_mix_before_weights true  --max_steps 12000 --save_checkpoints_on_s3

# Running small datasets for maximum 9K.
# DATASET="mrpc" # rte, mrpc, cola, stsb, wnli
#python -m torch.distributed.launch --nproc_per_node 4 run_glue.py  --dataset_name ${DATASET} ${shared_params} --output_dir $BASE_DIR"outputs/paper_experiments/glue_results/ours_self_condition_mean_mix_before_weights_"${DATASET}"_steps_"${num_inference_diffusion_steps}"_no_wd_max_steps_9k_for_small_data"  --num_inference_diffusion_steps ${num_inference_diffusion_steps} ${PARAMS_FOR_LOCAL} --weight_decay 0.0 --self_condition "logits_mean"  --per_device_train_batch_size 32  --gradient_accumulation_steps 1  --self_condition_mix_before_weights true  --max_steps 9000 --save_checkpoints_on_s3

# Running small datasets for 6K. => was not good.
#DATASET="wnli" # rte, mrpc, cola, stsb, wnli
#python -m torch.distributed.launch --nproc_per_node 4 run_glue.py  --dataset_name ${DATASET} ${shared_params} --output_dir $BASE_DIR"outputs/paper_experiments/glue_results/ours_self_condition_mean_mix_before_weights_"${DATASET}"_steps_"${num_inference_diffusion_steps}"_no_wd_max_steps_6k_for_small_data"  --num_inference_diffusion_steps ${num_inference_diffusion_steps} ${PARAMS_FOR_LOCAL} --weight_decay 0.0 --self_condition "logits_mean"  --per_device_train_batch_size 32  --gradient_accumulation_steps 1  --self_condition_mix_before_weights true  --max_steps 6000 --save_checkpoints_on_s3

# Running small data on 16K steps.
DATASET="rte" # rte, mrpc, cola, stsb, wnli
python -m torch.distributed.launch --nproc_per_node 4 run_glue.py  --dataset_name ${DATASET} ${shared_params} --output_dir $BASE_DIR"outputs/paper_experiments/glue_results/ours_self_condition_mean_mix_before_weights_"${DATASET}"_steps_"${num_inference_diffusion_steps}"_no_wd_max_16k_steps"  --num_inference_diffusion_steps ${num_inference_diffusion_steps} ${PARAMS_FOR_LOCAL} --weight_decay 0.0 --self_condition "logits_mean"  --per_device_train_batch_size 32  --gradient_accumulation_steps 1  --self_condition_mix_before_weights true  --max_steps 16000 --save_checkpoints_on_s3






# DBEUG
#python -m torch.distributed.launch --nproc_per_node 2  run_glue.py  --dataset_name ${DATASET} ${shared_params} --output_dir $BASE_DIR"outputs/debug"  --num_inference_diffusion_steps ${num_inference_diffusion_steps} ${PARAMS_FOR_LOCAL} --weight_decay 0.0 --self_condition "logits_mean"  --per_device_train_batch_size 32  --gradient_accumulation_steps 1  --self_condition_mix_before_weights true  --max_steps 25000 --save_checkpoints_on_s3  --max_steps 6 --save_steps 2 --eval_steps 2 --max_eval_samples 6 

