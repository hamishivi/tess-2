
# GLUE should be run with 128+label_length, where label_length=5.
shared_params="--model_name_or_path roberta-large --do_eval --do_predict --max_seq_length 133 --per_device_train_batch_size 64 --per_device_eval_batch_size 64 --evaluation_strategy steps --save_strategy steps   --report_to tensorboard  --pad_to_max_length  --simplex_value 5 --num_diffusion_steps 5000 --conditional_generation seq2seq  --learning_rate 3e-5 --gradient_accumulation_steps 2 --lr_scheduler_type linear --beta_schedule squaredcos_improved_ddpm  --top_p 0.99 --warmup_steps 500 --logging_steps 50 --save_steps 1000  --add_t5_tags --max_steps 100000   --load_best_model_at_end true --checkpoint_best_model  --greater_is_better true --eval_steps 1000 --load_states_in_eval_from_model_path"
shared_params_without_top_p="--model_name_or_path roberta-large --do_eval --do_predict --max_seq_length 133 --per_device_train_batch_size 64 --per_device_eval_batch_size 64 --evaluation_strategy steps --save_strategy steps   --report_to tensorboard  --pad_to_max_length  --simplex_value 5 --num_diffusion_steps 5000 --conditional_generation seq2seq  --learning_rate 3e-5 --gradient_accumulation_steps 2 --lr_scheduler_type linear --beta_schedule squaredcos_improved_ddpm   --warmup_steps 500 --logging_steps 50 --save_steps 1000  --add_t5_tags --max_steps 100000   --load_best_model_at_end true --checkpoint_best_model  --greater_is_better true --eval_steps 1000 --load_states_in_eval_from_model_path"

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
'

########################################

# ****** this is selected *********
# eval for the model with max_steps_set.
: '
DATASETS=("mrpc"    "rte"  "stsb"  "wnli"  "qqp"   "qnli"    "sst2" "mnli" "cola") 
CHECKPOINTS=("7000"    "3000" "4000"  "8000"  "23000"  "19000" "18000" "14000" "6000")
for i in "${!DATASETS[@]}"; do
    DATASET=${DATASETS[i]}
    CHECKPOINT=${CHECKPOINTS[i]}
    output_dir=${BASE_DIR}"outputs/paper_experiments/glue_results/ours_self_condition_mean_mix_before_weights_"${DATASET}"_steps_10_no_wd_max_steps_set/" 
    model_name_or_path=${output_dir}"/checkpoint-"${CHECKPOINT}
    for TOP_P in 0.9 0.95 0.99 
    do	    
    python  -m torch.distributed.launch --nproc_per_node 8  run_glue.py  --dataset_name ${DATASETS[i]} ${shared_params_without_top_p} --output_dir ${output_dir}  --num_inference_diffusion_steps ${num_inference_diffusion_steps} ${PARAMS_FOR_LOCAL} --weight_decay 0.0 --model_name_or_path ${model_name_or_path} --self_condition "logits_mean" --self_condition_mix_before_weights true --top_p ${TOP_P}
    done
    # run with top-p=None
    python  -m torch.distributed.launch --nproc_per_node 8  run_glue.py  --dataset_name ${DATASETS[i]} ${shared_params_without_top_p} --output_dir ${output_dir}  --num_inference_diffusion_steps ${num_inference_diffusion_steps} ${PARAMS_FOR_LOCAL} --weight_decay 0.0 --model_name_or_path ${model_name_or_path} --self_condition "logits_mean" --self_condition_mix_before_weights true

done
'

# evaluate mrpc for different number of steps.
DATASET="mrpc"
model_path=${BASE_DIR}"outputs/paper_experiments/glue_results/ours_self_condition_mean_mix_before_weights_"${DATASET}"_steps_10_no_wd_max_steps_set/checkpoint-7000" 
num_inference_diffusion_steps=10
python  -m torch.distributed.launch --nproc_per_node 8  run_glue.py  --dataset_name ${DATASET} ${shared_params_without_top_p} --output_dir ${model_path}"/inference_ablation/step_"${num_inference_diffusion_steps}  --num_inference_diffusion_steps ${num_inference_diffusion_steps} ${PARAMS_FOR_LOCAL} --weight_decay 0.0 --model_name_or_path ${model_path} --self_condition "logits_mean" --self_condition_mix_before_weights true   --max_predict_samples 1000



: '
# evaluate for small data for max-steps 6k.
DATASETS=("mrpc"    "rte"  "stsb"  "wnli" "cola") 
CHECKPOINTS=("2000" "3000" "4000" "2000" "5000")
for i in "${!DATASETS[@]}"; do
    DATASET=${DATASETS[i]}
    CHECKPOINT=${CHECKPOINTS[i]}
    output_dir=$BASE_DIR"outputs/paper_experiments/glue_results/ours_self_condition_mean_mix_before_weights_"${DATASET}"_steps_"${num_inference_diffusion_steps}"_no_wd_max_steps_6k_for_small_data/"
    model_name_or_path=$output_dir"/checkpoint-"${CHECKPOINT} 
    python  -m torch.distributed.launch --nproc_per_node 8  run_glue.py  --dataset_name ${DATASETS[i]} ${shared_params} --output_dir ${output_dir}  --num_inference_diffusion_steps ${num_inference_diffusion_steps} ${PARAMS_FOR_LOCAL} --weight_decay 0.0 --model_name_or_path ${model_name_or_path} --self_condition "logits_mean" --self_condition_mix_before_weights true
done

# evaluate for small data for max-steps 9k.
DATASETS=("mrpc"    "rte"  "stsb"  "wnli" "cola") 
CHECKPOINTS=("1000"  "3000" "3000" "7000" "2000")
for i in "${!DATASETS[@]}"; do
    DATASET=${DATASETS[i]}
    CHECKPOINT=${CHECKPOINTS[i]}
    output_dir=$BASE_DIR"outputs/paper_experiments/glue_results/ours_self_condition_mean_mix_before_weights_"${DATASET}"_steps_"${num_inference_diffusion_steps}"_no_wd_max_steps_9k_for_small_data"
    model_name_or_path=${output_dir}"/checkpoint-"${CHECKPOINT} 
    python  -m torch.distributed.launch --nproc_per_node 8  run_glue.py  --dataset_name ${DATASETS[i]} ${shared_params} --output_dir ${output_dir}  --num_inference_diffusion_steps ${num_inference_diffusion_steps} ${PARAMS_FOR_LOCAL} --weight_decay 0.0 --model_name_or_path ${model_name_or_path} --self_condition "logits_mean" --self_condition_mix_before_weights true
done

# evaluate for small data for max-steps 16k.
DATASETS=("mrpc"    "rte"  "stsb"  "wnli" "cola") 
CHECKPOINTS=("8000" "3000" "7000" "9000" "8000")
for i in "${!DATASETS[@]}"; do
    DATASET=${DATASETS[i]}
    CHECKPOINT=${CHECKPOINTS[i]}
    output_dir=$BASE_DIR"outputs/paper_experiments/glue_results/ours_self_condition_mean_mix_before_weights_"${DATASET}"_steps_"${num_inference_diffusion_steps}"_no_wd_max_16k_steps"
    model_name_or_path=${output_dir}"/checkpoint-"${CHECKPOINT} 
    python  -m torch.distributed.launch --nproc_per_node 8  run_glue.py  --dataset_name ${DATASETS[i]} ${shared_params} --output_dir ${output_dir}  --num_inference_diffusion_steps ${num_inference_diffusion_steps} ${PARAMS_FOR_LOCAL} --weight_decay 0.0 --model_name_or_path ${model_name_or_path} --self_condition "logits_mean" --self_condition_mix_before_weights true
done


# evaluate the baseline model.
DATASETS=("mrpc"    "rte"  "stsb"  "wnli"  "qqp"   "qnli"    "sst2"  "mnli" "cola") 
CHECKPOINTS=("290"    "190" "270"  "5"     "8505"  "2430"     "1036" "6136" "469")
for i in "${!DATASETS[@]}"; do
    DATASET=${DATASETS[i]}
    CHECKPOINT=${CHECKPOINTS[i]}
    output_dir="/net/nfs.cirrascale/s2-research/rabeehk/outputs/paper_experiments/glue_results/baseline_"${DATASET} 
    model_name_or_path=${output_dir}"/checkpoint-"${CHECKPOINT} 
    python baselines/run_glue.py --model_name_or_path ${model_name_or_path}  --dataset_name ${DATASET} --do_eval --do_predict --max_seq_length 128 --per_device_train_batch_size 64 --per_device_eval_batch_size 64 --evaluation_strategy epoch --save_strategy epoch  --output_dir ${output_dir} --report_to tensorboard  --overwrite_output_dir --pad_to_max_length --learning_rate 3e-5 --num_train_epochs 3 --logging_steps 50  --load_best_model_at_end true --checkpoint_best_model --greater_is_better true --warmup_steps 500  --tokenizer_name roberta-large --save_total_limit 1 --lr_scheduler_type linear  --gradient_accumulation_steps 2 --load_states_in_eval_from_model_path
done


# cola with different seed.
DATASETS=("cola"    "cola" "cola") 
CHECKPOINTS=()
SEEDS=()
for i in "${!DATASETS[@]}"; do
    DATASET=${DATASETS[i]}
    CHECKPOINT=${CHECKPOINTS[i]}
    seed=${SEEDS[i]}
    output_dir=$BASE_DIR"outputs/paper_experiments/glue_results/ours_self_condition_mean_mix_before_weights_"${DATASET}"_steps_"${num_inference_diffusion_steps}"_no_wd_max_16k_steps_seed_"${seed}
    model_name_or_path=${output_dir}"/checkpoint-"${CHECKPOINT} 
    python  -m torch.distributed.launch --nproc_per_node 8  run_glue.py  --dataset_name ${DATASETS[i]} ${shared_params} --output_dir ${output_dir}  --num_inference_diffusion_steps ${num_inference_diffusion_steps} ${PARAMS_FOR_LOCAL} --weight_decay 0.0 --model_name_or_path ${model_name_or_path} --self_condition "logits_mean" --self_condition_mix_before_weights true
done
'

