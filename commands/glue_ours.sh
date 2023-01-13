
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
#python run_glue.py  --dataset_name ${DATASET} ${shared_params} --output_dir $BASE_DIR"outputs/paper_experiments/ours_glue/debug"  --num_inference_diffusion_steps ${num_inference_diffusion_steps} ${PARAMS_FOR_LOCAL} ${DEBUG_PARAMS} 


# Training GLUE with self-conditioning mean for now.
DATASET="mrpc"
python run_glue.py  --dataset_name ${DATASET} ${shared_params} --output_dir $BASE_DIR"outputs/paper_experiments/ours_glue_self_condition_mean/"${DATASET}"_steps_"${num_inference_diffusion_steps}"_wd_0.01"  --num_inference_diffusion_steps ${num_inference_diffusion_steps} ${PARAMS_FOR_LOCAL} --weight_decay 0.01 --self_condition "logits_mean" --per_device_train_batch_size 32  --gradient_accumulation_steps 4


