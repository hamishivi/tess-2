
# GLUE should be run with 128+label_length, where label_length=5.
shared_params="--model_name_or_path roberta-large --do_eval --do_predict --max_seq_length 133 --per_device_train_batch_size 64 --per_device_eval_batch_size 64 --evaluation_strategy steps --save_strategy steps   --report_to tensorboard  --pad_to_max_length  --simplex_value 5 --num_diffusion_steps 5000 --conditional_generation seq2seq  --learning_rate 3e-5 --gradient_accumulation_steps 2 --lr_scheduler_type linear --beta_schedule squaredcos_improved_ddpm  --top_p 0.99 --warmup_steps 500 --logging_steps 50 --save_steps 1000  --add_t5_tags --max_steps 100000   --load_best_model_at_end true --checkpoint_best_model  --greater_is_better true --eval_steps 1000 --load_states_in_eval_from_model_path"
BASE_DIR="/net/nfs.cirrascale/s2-research/rabeehk/"
PARAMS_FOR_LOCAL=" --save_total_limit 1"
DEBUG_PARAMS="--eval_steps 2 --num_inference_diffusion_steps 3 --per_device_train_batch_size 12 --max_eval_samples 6"
num_inference_diffusion_steps=10


: '
DATASET="cola"
model_name_or_path="/net/nfs.cirrascale/s2-research/rabeehk/outputs/paper_experiments/ours_glue/cola_steps_10_wd_0.01/checkpoint-75000"
python run_glue.py  --dataset_name ${DATASET} ${shared_params} --output_dir $BASE_DIR"outputs/paper_experiments/ours_glue/"${DATASET}"_steps_"${num_inference_diffusion_steps}"_wd_0.01"  --num_inference_diffusion_steps ${num_inference_diffusion_steps} ${PARAMS_FOR_LOCAL} --weight_decay 0.01 --model_name_or_path ${model_name_or_path}


DATASET="mrpc"
model_name_or_path="/net/nfs.cirrascale/s2-research/rabeehk/outputs/paper_experiments/ours_glue/mrpc_steps_10_wd_0.01/checkpoint-15000/"
python run_glue.py  --dataset_name ${DATASET} ${shared_params} --output_dir $BASE_DIR"outputs/paper_experiments/ours_glue/"${DATASET}"_steps_"${num_inference_diffusion_steps}"_wd_0.01"  --num_inference_diffusion_steps ${num_inference_diffusion_steps} ${PARAMS_FOR_LOCAL} --weight_decay 0.01 --model_name_or_path ${model_name_or_path}


DATASET="rte"
model_name_or_path="/net/nfs.cirrascale/s2-research/rabeehk/outputs/paper_experiments/ours_glue/rte_steps_10_wd_0.01/checkpoint-75000/"
python run_glue.py  --dataset_name ${DATASET} ${shared_params} --output_dir $BASE_DIR"outputs/paper_experiments/ours_glue/"${DATASET}"_steps_"${num_inference_diffusion_steps}"_wd_0.01"  --num_inference_diffusion_steps ${num_inference_diffusion_steps} ${PARAMS_FOR_LOCAL} --weight_decay 0.01 --model_name_or_path ${model_name_or_path}


DATASET="stsb"
model_name_or_path="/net/nfs.cirrascale/s2-research/rabeehk/outputs/paper_experiments/ours_glue/stsb_steps_10_wd_0.01/checkpoint-73000/"
python run_glue.py  --dataset_name ${DATASET} ${shared_params} --output_dir $BASE_DIR"outputs/paper_experiments/ours_glue/"${DATASET}"_steps_"${num_inference_diffusion_steps}"_wd_0.01"  --num_inference_diffusion_steps ${num_inference_diffusion_steps} ${PARAMS_FOR_LOCAL} --weight_decay 0.01 --model_name_or_path ${model_name_or_path}

DATASET="wnli"
model_name_or_path="/net/nfs.cirrascale/s2-research/rabeehk/outputs/paper_experiments/ours_glue/wnli_steps_10_wd_0.01/checkpoint-76000/"
python run_glue.py  --dataset_name ${DATASET} ${shared_params} --output_dir $BASE_DIR"outputs/paper_experiments/ours_glue/"${DATASET}"_steps_"${num_inference_diffusion_steps}"_wd_0.01"  --num_inference_diffusion_steps ${num_inference_diffusion_steps} ${PARAMS_FOR_LOCAL} --weight_decay 0.01 --model_name_or_path ${model_name_or_path}

DATASET="qqp"
model_name_or_path="/net/nfs.cirrascale/s2-research/rabeehk/outputs/paper_experiments/ours_glue/qqp_steps_10_wd_0.01_copied/checkpoint-72000/"
python -m torch.distributed.launch --nproc_per_node 4  run_glue.py  --dataset_name ${DATASET} ${shared_params} --output_dir $BASE_DIR"outputs/paper_experiments/ours_glue/"${DATASET}"_steps_"${num_inference_diffusion_steps}"_wd_0.01"  --num_inference_diffusion_steps ${num_inference_diffusion_steps} ${PARAMS_FOR_LOCAL} --weight_decay 0.01 --model_name_or_path ${model_name_or_path}


DATASET="qnli"
model_name_or_path="/net/nfs.cirrascale/s2-research/rabeehk/outputs/paper_experiments/ours_glue/qnli_steps_10_wd_0.01_copied/checkpoint-73000"
python -m torch.distributed.launch --nproc_per_node 4  run_glue.py  --dataset_name ${DATASET} ${shared_params} --output_dir $BASE_DIR"outputs/paper_experiments/ours_glue/"${DATASET}"_steps_"${num_inference_diffusion_steps}"_wd_0.01"  --num_inference_diffusion_steps ${num_inference_diffusion_steps} ${PARAMS_FOR_LOCAL} --weight_decay 0.01 --model_name_or_path ${model_name_or_path}

DATASET="sst2"
model_name_or_path="/net/nfs.cirrascale/s2-research/rabeehk/outputs/paper_experiments/ours_glue/sst2_steps_10_wd_0.01_copied/checkpoint-74000"
python -m torch.distributed.launch --nproc_per_node 4  run_glue.py  --dataset_name ${DATASET} ${shared_params} --output_dir $BASE_DIR"outputs/paper_experiments/ours_glue/"${DATASET}"_steps_"${num_inference_diffusion_steps}"_wd_0.01"  --num_inference_diffusion_steps ${num_inference_diffusion_steps} ${PARAMS_FOR_LOCAL} --weight_decay 0.01 --model_name_or_path ${model_name_or_path}

DATASET="mnli"
model_name_or_path="/net/nfs.cirrascale/s2-research/rabeehk/outputs/paper_experiments/ours_glue/mnli_steps_10_wd_0.01_copied/checkpoint-55000"
python -m torch.distributed.launch --nproc_per_node 4  run_glue.py  --dataset_name ${DATASET} ${shared_params} --output_dir $BASE_DIR"outputs/paper_experiments/ours_glue/"${DATASET}"_steps_"${num_inference_diffusion_steps}"_wd_0.01"  --num_inference_diffusion_steps ${num_inference_diffusion_steps} ${PARAMS_FOR_LOCAL} --weight_decay 0.01 --model_name_or_path ${model_name_or_path}
'

: '
DATASETS=("mrpc") #, "rte" "stsb"  "wnli"  "qqp"   "qnli" "sst2" "mnli", "cola") 
CHECKPOINTS=("8000") #, "2000" "2000" "10000" "43000" "3000" "9000" "9000", "6000")
for i in "${!DATASETS[@]}"; do
    DATASET=${DATASETS[i]}
    CHECKPOINT=${CHECKPOINTS[i]}
    model_name_or_path="/net/nfs.cirrascale/s2-research/rabeehk/outputs/paper_experiments/ours_glue_self_condition_mean/"${DATASET}"_steps_10_wd_0.01/checkpoint-"${CHECKPOINT}
    python run_glue.py  --dataset_name ${DATASETS[i]} ${shared_params} --output_dir $BASE_DIR"outputs/paper_experiments/ours_glue_self_condition_mean/"${DATASET}"_steps_"${num_inference_diffusion_steps}"_wd_0.01"  --num_inference_diffusion_steps ${num_inference_diffusion_steps} ${PARAMS_FOR_LOCAL} --weight_decay 0.01 --model_name_or_path ${model_name_or_path}
done
    
# evaluate the models trained from a checkpoint.
DATASETS=("mrpc"    "rte"  "stsb"  "wnli"  "qqp"   "qnli" "sst2" "mnli" "cola") 
CHECKPOINTS=("6000" "1000" "6000" "1000"  "14000" "7000" "9000" "10000" "4000")
for i in "${!DATASETS[@]}"; do
    DATASET=${DATASETS[i]}
    CHECKPOINT=${CHECKPOINTS[i]}
    model_name_or_path="/net/nfs.cirrascale/s2-research/rabeehk/outputs/paper_experiments/ours_glue_self_condition_mean/"${DATASET}"_steps_10_wd_0.01_from_40K_checkpoint/checkpoint-"${CHECKPOINT}
    python run_glue.py  --dataset_name ${DATASETS[i]} ${shared_params} --output_dir $BASE_DIR"outputs/paper_experiments/ours_glue_self_condition_mean/"${DATASET}"_steps_"${num_inference_diffusion_steps}"_wd_0.01_from_40K_checkpoint"  --num_inference_diffusion_steps ${num_inference_diffusion_steps} ${PARAMS_FOR_LOCAL} --weight_decay 0.01 --model_name_or_path ${model_name_or_path}
done
'


:'
# evaluate our models without a wd.
DATASETS=("mrpc"    "rte"  "stsb"  "wnli"  "qqp"   "qnli" "sst2" "mnli" "cola") 
CHECKPOINTS=("4000" "11000" "2000" "12000" "15000" "3000" "5000" "12000" "1000")
for i in "${!DATASETS[@]}"; do
    DATASET=${DATASETS[i]}
    CHECKPOINT=${CHECKPOINTS[i]}
    model_name_or_path=$BASE_DIR"outputs/paper_experiments/glue_results/ours_self_condition_mean_mix_before_weights_"${DATASET}"_steps_10_no_wd/checkpoint-"${CHECKPOINT} 
    python run_glue.py  --dataset_name ${DATASETS[i]} ${shared_params} --output_dir $BASE_DIR"outputs/paper_experiments/glue_results/ours_self_condition_mean_mix_before_weights_"${DATASET}"_steps_"${num_inference_diffusion_steps}"_no_wd"  --num_inference_diffusion_steps ${num_inference_diffusion_steps} ${PARAMS_FOR_LOCAL} --weight_decay 0.0 --model_name_or_path ${model_name_or_path} --self_condition "logits_mean" --self_condition_mix_before_weights true
done
'

# evaluate our models with a wd.
: '
DATASETS=("mrpc"    "rte"  "stsb"  "wnli"  "qqp"   "qnli" "sst2" "mnli" "cola") 
CHECKPOINTS=("6000" "2000" "2000" "9000" "7000"  "11000" "14000" "5000" "4000")
for i in "${!DATASETS[@]}"; do
    DATASET=${DATASETS[i]}
    CHECKPOINT=${CHECKPOINTS[i]}
    model_name_or_path=$BASE_DIR"outputs/paper_experiments/glue_results/ours_self_condition_mean_mix_before_weights_"${DATASET}"_steps_10_wd_0.01/checkpoint-"${CHECKPOINT} 
    python run_glue.py  --dataset_name ${DATASETS[i]} ${shared_params} --output_dir $BASE_DIR"outputs/paper_experiments/glue_results/ours_self_condition_mean_mix_before_weights_"${DATASET}"_steps_10_wd_0.01/"  --num_inference_diffusion_steps ${num_inference_diffusion_steps} ${PARAMS_FOR_LOCAL} --weight_decay 0.01 --model_name_or_path ${model_name_or_path} --self_condition "logits_mean" --self_condition_mix_before_weights true
done

# does more steps help?
# new for glue
DATASETS=( "qqp"    "sst2" ) 
CHECKPOINTS=("21000"  "22000")
for i in "${!DATASETS[@]}"; do
    DATASET=${DATASETS[i]}
    CHECKPOINT=${CHECKPOINTS[i]}
    model_name_or_path=$BASE_DIR"outputs/paper_experiments/glue_results/ours_self_condition_mean_mix_before_weights_"${DATASET}"_steps_10_no_wd/checkpoint-"${CHECKPOINT} 
    python run_glue.py  --dataset_name ${DATASETS[i]} ${shared_params} --output_dir $BASE_DIR"outputs/paper_experiments/glue_results/ours_self_condition_mean_mix_before_weights_"${DATASET}"_steps_"${num_inference_diffusion_steps}"_no_wd"  --num_inference_diffusion_steps ${num_inference_diffusion_steps} ${PARAMS_FOR_LOCAL} --weight_decay 0.0 --model_name_or_path ${model_name_or_path} --self_condition "logits_mean" --self_condition_mix_before_weights true
done
