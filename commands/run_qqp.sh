# trains on the qqp datasets for paraphrase generation.
PARAMS_FOR_LOCAL=" --save_total_limit 1 "

learning_rate=3e-5
num_inference_diffusion_steps=1000
max_steps=90000 #50000 # We try for 30000 and 100000
#python -m torch.distributed.launch --nproc_per_node 8 run_simplification.py --model_name_or_path "roberta-base" --do_train --do_eval --do_predict --dataset_name qqp  --output_dir  "/net/nfs.cirrascale/s2-research/rabeehk/outputs/paper_experiments/qqp_tune_steps/ours_lr_"${learning_rate}"_steps_"${max_steps} --per_device_train_batch_size=1 --per_device_eval_batch_size=12   --report_to tensorboard --eval_steps 1000  --max_steps ${max_steps} --max_source_length 100  --max_target_length 85 --max_seq_length 185 --conditional_generation "seq2seq" --num_inference_diffusion_steps ${num_inference_diffusion_steps} --evaluation_strategy steps --simplex_value 5 --num_diffusion_steps 5000 --lr_scheduler_type linear --learning_rate ${learning_rate} --pad_to_max_length  --beta_schedule squaredcos_improved_ddpm --warmup_steps 2000 --logging_steps 50 --save_steps 1000    --self_condition "logits_mean"  --self_condition_mix_before_weights true --load_states_in_eval_from_model_path true --max_eval_samples 96 ${PARAMS_FOR_LOCAL} --weight_decay 0.0 --save_checkpoints_on_s3 --dataset_folder "/net/nfs.cirrascale/s2-research/rabeehk/simplex-diffusion/datasets/qqp/"

# **** this is reported ****
: '
# evaluate it
for max_steps in 50000 90000
do
model_path="/net/nfs.cirrascale/s2-research/rabeehk/outputs/paper_experiments/qqp_tune_steps/ours_lr_"${learning_rate}"_steps_"${max_steps}"/checkpoint-"${max_steps}
python -m torch.distributed.launch --nproc_per_node 8 run_simplification.py --model_name_or_path ${model_path}  --do_eval --do_predict --dataset_name qqp  --output_dir  ${model_path} --per_device_train_batch_size=1 --per_device_eval_batch_size=12   --report_to tensorboard --eval_steps 1000  --max_steps ${max_steps} --max_source_length 100  --max_target_length 85 --max_seq_length 185 --conditional_generation "seq2seq" --num_inference_diffusion_steps ${num_inference_diffusion_steps} --evaluation_strategy steps --simplex_value 5 --num_diffusion_steps 5000 --lr_scheduler_type linear --learning_rate ${learning_rate} --pad_to_max_length  --beta_schedule squaredcos_improved_ddpm --warmup_steps 2000 --logging_steps 50 --save_steps 1000    --self_condition "logits_mean"  --self_condition_mix_before_weights true --load_states_in_eval_from_model_path true --max_eval_samples 96 ${PARAMS_FOR_LOCAL} --weight_decay 0.0 --save_checkpoints_on_s3 --dataset_folder "/net/nfs.cirrascale/s2-research/rabeehk/simplex-diffusion/datasets/qqp/" --load_states_in_eval_from_model_path
done 
'

: '
# eval qqp for 10 seeds.
max_steps=90000
learning_rate=3e-5
#for seed in 58 95 5 
#for seed in 60 27 74
for seed in 21 84 36 
do  
model_path="/net/nfs.cirrascale/s2-research/rabeehk/outputs/paper_experiments/qqp_tune_steps/ours_lr_"${learning_rate}"_steps_"${max_steps}"/checkpoint-"${max_steps}
python -m torch.distributed.launch --nproc_per_node 8 run_simplification.py --model_name_or_path ${model_path}  --do_eval --do_predict --dataset_name qqp  --output_dir  ${model_path}"/seeds_10" --per_device_train_batch_size=1 --per_device_eval_batch_size=12   --report_to tensorboard --eval_steps 1000  --max_steps ${max_steps} --max_source_length 100  --max_target_length 85 --max_seq_length 185 --conditional_generation "seq2seq" --num_inference_diffusion_steps ${num_inference_diffusion_steps} --evaluation_strategy steps --simplex_value 5 --num_diffusion_steps 5000 --lr_scheduler_type linear --learning_rate ${learning_rate} --pad_to_max_length  --beta_schedule squaredcos_improved_ddpm --warmup_steps 2000 --logging_steps 50 --save_steps 1000    --self_condition "logits_mean"  --self_condition_mix_before_weights true --load_states_in_eval_from_model_path true --max_eval_samples 96 ${PARAMS_FOR_LOCAL} --weight_decay 0.0 --save_checkpoints_on_s3 --dataset_folder "/net/nfs.cirrascale/s2-research/rabeehk/simplex-diffusion/datasets/qqp/" --load_states_in_eval_from_model_path --seed ${seed}  --generate_with_seed true
done 
'

# Ablation for self-condition.
learning_rate=3e-5
num_inference_diffusion_steps=1000
max_steps=60000 
#python -m torch.distributed.launch --nproc_per_node 8 run_simplification.py --model_name_or_path "roberta-base" --do_train --do_eval --do_predict --dataset_name qqp  --output_dir  "/net/nfs.cirrascale/s2-research/rabeehk/outputs/paper_experiments/qqp_self_condition_ablations/ours_lr_"${learning_rate}"_steps_"${max_steps}"_self_condition_logits_mean_mix_before_weights" --per_device_train_batch_size=1 --per_device_eval_batch_size=12   --report_to tensorboard --eval_steps 1000  --max_steps ${max_steps} --max_source_length 100  --max_target_length 85 --max_seq_length 185 --conditional_generation "seq2seq" --num_inference_diffusion_steps ${num_inference_diffusion_steps} --evaluation_strategy steps --simplex_value 5 --num_diffusion_steps 5000 --lr_scheduler_type linear --learning_rate ${learning_rate} --pad_to_max_length  --beta_schedule squaredcos_improved_ddpm --warmup_steps 2000 --logging_steps 50 --save_steps 1000    --self_condition "logits_mean"  --self_condition_mix_before_weights true --load_states_in_eval_from_model_path true --max_eval_samples 96 ${PARAMS_FOR_LOCAL} --weight_decay 0.0 --save_checkpoints_on_s3 --dataset_folder "/net/nfs.cirrascale/s2-research/rabeehk/simplex-diffusion/datasets/qqp/"
#python -m torch.distributed.launch --nproc_per_node 8 run_simplification.py --model_name_or_path "roberta-base" --do_train --do_eval --do_predict --dataset_name qqp  --output_dir  "/net/nfs.cirrascale/s2-research/rabeehk/outputs/paper_experiments/qqp_self_condition_ablations/ours_lr_"${learning_rate}"_steps_"${max_steps}"_self_logits" --per_device_train_batch_size=1 --per_device_eval_batch_size=12   --report_to tensorboard --eval_steps 1000  --max_steps ${max_steps} --max_source_length 100  --max_target_length 85 --max_seq_length 185 --conditional_generation "seq2seq" --num_inference_diffusion_steps ${num_inference_diffusion_steps} --evaluation_strategy steps --simplex_value 5 --num_diffusion_steps 5000 --lr_scheduler_type linear --learning_rate ${learning_rate} --pad_to_max_length  --beta_schedule squaredcos_improved_ddpm --warmup_steps 2000 --logging_steps 50 --save_steps 1000    --self_condition "logits"  --load_states_in_eval_from_model_path true --max_eval_samples 96 ${PARAMS_FOR_LOCAL} --weight_decay 0.0 --save_checkpoints_on_s3 --dataset_folder "/net/nfs.cirrascale/s2-research/rabeehk/simplex-diffusion/datasets/qqp/"
#python -m torch.distributed.launch --nproc_per_node 8 run_simplification.py --model_name_or_path "roberta-base" --do_train --do_eval --do_predict --dataset_name qqp  --output_dir  "/net/nfs.cirrascale/s2-research/rabeehk/outputs/paper_experiments/qqp_self_condition_ablations/ours_lr_"${learning_rate}"_steps_"${max_steps}"_no_self" --per_device_train_batch_size=1 --per_device_eval_batch_size=12   --report_to tensorboard --eval_steps 1000  --max_steps ${max_steps} --max_source_length 100  --max_target_length 85 --max_seq_length 185 --conditional_generation "seq2seq" --num_inference_diffusion_steps ${num_inference_diffusion_steps} --evaluation_strategy steps --simplex_value 5 --num_diffusion_steps 5000 --lr_scheduler_type linear --learning_rate ${learning_rate} --pad_to_max_length  --beta_schedule squaredcos_improved_ddpm --warmup_steps 2000 --logging_steps 50 --save_steps 1000    --load_states_in_eval_from_model_path true --max_eval_samples 96 ${PARAMS_FOR_LOCAL} --weight_decay 0.0 --save_checkpoints_on_s3 --dataset_folder "/net/nfs.cirrascale/s2-research/rabeehk/simplex-diffusion/datasets/qqp/"

# resume aboves after cluster crash.
#model_path="/net/nfs.cirrascale/s2-research/rabeehk/outputs/paper_experiments/qqp_self_condition_ablations/ours_lr_"${learning_rate}"_steps_"${max_steps}"_no_self/checkpoint-60000/"
#python -m torch.distributed.launch --nproc_per_node 8 run_simplification.py --model_name_or_path ${model_path} --do_eval --do_predict --dataset_name qqp  --output_dir ${model_path}  --per_device_train_batch_size=1 --per_device_eval_batch_size=12   --report_to tensorboard --eval_steps 1000  --max_steps ${max_steps} --max_source_length 100  --max_target_length 85 --max_seq_length 185 --conditional_generation "seq2seq" --num_inference_diffusion_steps ${num_inference_diffusion_steps} --evaluation_strategy steps --simplex_value 5 --num_diffusion_steps 5000 --lr_scheduler_type linear --learning_rate ${learning_rate} --pad_to_max_length  --beta_schedule squaredcos_improved_ddpm --warmup_steps 2000 --logging_steps 50 --save_steps 1000    --load_states_in_eval_from_model_path true --max_eval_samples 96 ${PARAMS_FOR_LOCAL} --weight_decay 0.0 --save_checkpoints_on_s3 --dataset_folder "/net/nfs.cirrascale/s2-research/rabeehk/simplex-diffusion/datasets/qqp/"

#checkpoint="/net/nfs.cirrascale/s2-research/rabeehk/outputs/paper_experiments/qqp_self_condition_ablations/ours_lr_"${learning_rate}"_steps_"${max_steps}"_self_logits/checkpoint-56000/"
#python -m torch.distributed.launch --nproc_per_node 8 run_simplification.py --model_name_or_path "roberta-base" --do_train --do_eval --do_predict --dataset_name qqp  --output_dir  "/net/nfs.cirrascale/s2-research/rabeehk/outputs/paper_experiments/qqp_self_condition_ablations/ours_lr_"${learning_rate}"_steps_"${max_steps}"_self_logits" --per_device_train_batch_size=1 --per_device_eval_batch_size=12   --report_to tensorboard --eval_steps 1000  --max_steps ${max_steps} --max_source_length 100  --max_target_length 85 --max_seq_length 185 --conditional_generation "seq2seq" --num_inference_diffusion_steps ${num_inference_diffusion_steps} --evaluation_strategy steps --simplex_value 5 --num_diffusion_steps 5000 --lr_scheduler_type linear --learning_rate ${learning_rate} --pad_to_max_length  --beta_schedule squaredcos_improved_ddpm --warmup_steps 2000 --logging_steps 50 --save_steps 1000    --self_condition "logits"  --load_states_in_eval_from_model_path true --max_eval_samples 96 ${PARAMS_FOR_LOCAL} --weight_decay 0.0 --save_checkpoints_on_s3 --dataset_folder "/net/nfs.cirrascale/s2-research/rabeehk/simplex-diffusion/datasets/qqp/" --resume_from_checkpoint ${checkpoint}

#checkpoint="/net/nfs.cirrascale/s2-research/rabeehk/outputs/paper_experiments/qqp_self_condition_ablations/ours_lr_"${learning_rate}"_steps_"${max_steps}"_self_condition_logits_mean_mix_before_weights/checkpoint-52000"
#python -m torch.distributed.launch --nproc_per_node 8 run_simplification.py --model_name_or_path "roberta-base" --do_train --do_eval --do_predict --dataset_name qqp  --output_dir  "/net/nfs.cirrascale/s2-research/rabeehk/outputs/paper_experiments/qqp_self_condition_ablations/ours_lr_"${learning_rate}"_steps_"${max_steps}"_self_condition_logits_mean_mix_before_weights" --per_device_train_batch_size=1 --per_device_eval_batch_size=12   --report_to tensorboard --eval_steps 1000  --max_steps ${max_steps} --max_source_length 100  --max_target_length 85 --max_seq_length 185 --conditional_generation "seq2seq" --num_inference_diffusion_steps ${num_inference_diffusion_steps} --evaluation_strategy steps --simplex_value 5 --num_diffusion_steps 5000 --lr_scheduler_type linear --learning_rate ${learning_rate} --pad_to_max_length  --beta_schedule squaredcos_improved_ddpm --warmup_steps 2000 --logging_steps 50 --save_steps 1000    --self_condition "logits_mean"  --self_condition_mix_before_weights true --load_states_in_eval_from_model_path true --max_eval_samples 96 ${PARAMS_FOR_LOCAL} --weight_decay 0.0 --save_checkpoints_on_s3 --dataset_folder "/net/nfs.cirrascale/s2-research/rabeehk/simplex-diffusion/datasets/qqp/"  --resume_from_checkpoint ${checkpoint}

# eval the above for different top-p
:'
for top_p in 0.9 0.95 0.99
do
	model_path="/net/nfs.cirrascale/s2-research/rabeehk/outputs/paper_experiments/qqp_self_condition_ablations/ours_lr_"${learning_rate}"_steps_"${max_steps}"_self_condition_logits_mean_mix_before_weights/checkpoint-60000"
python -m torch.distributed.launch --nproc_per_node 8 run_simplification.py --model_name_or_path ${model_path} --do_eval --do_predict --dataset_name qqp  --output_dir ${model_path} --per_device_train_batch_size=1 --per_device_eval_batch_size=12   --report_to tensorboard --eval_steps 1000  --max_steps ${max_steps} --max_source_length 100  --max_target_length 85 --max_seq_length 185 --conditional_generation "seq2seq" --num_inference_diffusion_steps ${num_inference_diffusion_steps} --evaluation_strategy steps --simplex_value 5 --num_diffusion_steps 5000 --lr_scheduler_type linear --learning_rate ${learning_rate} --pad_to_max_length  --beta_schedule squaredcos_improved_ddpm --warmup_steps 2000 --logging_steps 50 --save_steps 1000    --self_condition "logits_mean"  --self_condition_mix_before_weights true --load_states_in_eval_from_model_path true --max_eval_samples 96 ${PARAMS_FOR_LOCAL} --weight_decay 0.0 --save_checkpoints_on_s3 --dataset_folder "/net/nfs.cirrascale/s2-research/rabeehk/simplex-diffusion/datasets/qqp/" --top_p ${top_p}
done
'

: '
for top_p in 0.9 0.95 0.99
do
	model_path="/net/nfs.cirrascale/s2-research/rabeehk/outputs/paper_experiments/qqp_self_condition_ablations/ours_lr_"${learning_rate}"_steps_"${max_steps}"_self_logits/checkpoint-60000"
python -m torch.distributed.launch --nproc_per_node 8 run_simplification.py --model_name_or_path ${model_path}  --do_eval --do_predict --dataset_name qqp  --output_dir  ${model_path} --per_device_train_batch_size=1 --per_device_eval_batch_size=12   --report_to tensorboard --eval_steps 1000  --max_steps ${max_steps} --max_source_length 100  --max_target_length 85 --max_seq_length 185 --conditional_generation "seq2seq" --num_inference_diffusion_steps ${num_inference_diffusion_steps} --evaluation_strategy steps --simplex_value 5 --num_diffusion_steps 5000 --lr_scheduler_type linear --learning_rate ${learning_rate} --pad_to_max_length  --beta_schedule squaredcos_improved_ddpm --warmup_steps 2000 --logging_steps 50 --save_steps 1000    --self_condition "logits"  --load_states_in_eval_from_model_path true --max_eval_samples 96 ${PARAMS_FOR_LOCAL} --weight_decay 0.0 --save_checkpoints_on_s3 --dataset_folder "/net/nfs.cirrascale/s2-research/rabeehk/simplex-diffusion/datasets/qqp/" --top_p ${top_p}
done

for top_p in 0.9 0.95 0.99
do
	model_path="/net/nfs.cirrascale/s2-research/rabeehk/outputs/paper_experiments/qqp_self_condition_ablations/ours_lr_"${learning_rate}"_steps_"${max_steps}"_no_self/checkpoint-60000"
python -m torch.distributed.launch --nproc_per_node 8 run_simplification.py --model_name_or_path ${model_path}  --do_eval --do_predict --dataset_name qqp  --output_dir  ${model_path} --per_device_train_batch_size=1 --per_device_eval_batch_size=12   --report_to tensorboard --eval_steps 1000  --max_steps ${max_steps} --max_source_length 100  --max_target_length 85 --max_seq_length 185 --conditional_generation "seq2seq" --num_inference_diffusion_steps ${num_inference_diffusion_steps} --evaluation_strategy steps --simplex_value 5 --num_diffusion_steps 5000 --lr_scheduler_type linear --learning_rate ${learning_rate} --pad_to_max_length  --beta_schedule squaredcos_improved_ddpm --warmup_steps 2000 --logging_steps 50 --save_steps 1000    --load_states_in_eval_from_model_path true --max_eval_samples 96 ${PARAMS_FOR_LOCAL} --weight_decay 0.0 --save_checkpoints_on_s3 --dataset_folder "/net/nfs.cirrascale/s2-research/rabeehk/simplex-diffusion/datasets/qqp/" --top_p ${top_p}
done
'

# eval qqp ablation for 40k

#model_path="/net/nfs.cirrascale/s2-research/rabeehk/outputs/paper_experiments/qqp_self_condition_ablations/ours_lr_"${learning_rate}"_steps_"${max_steps}"_no_self/checkpoint-40000/"
#python -m torch.distributed.launch --nproc_per_node 8 run_simplification.py --model_name_or_path ${model_path} --do_eval --do_predict --dataset_name qqp  --output_dir ${model_path}  --per_device_train_batch_size=1 --per_device_eval_batch_size=12   --report_to tensorboard --eval_steps 1000  --max_steps ${max_steps} --max_source_length 100  --max_target_length 85 --max_seq_length 185 --conditional_generation "seq2seq" --num_inference_diffusion_steps ${num_inference_diffusion_steps} --evaluation_strategy steps --simplex_value 5 --num_diffusion_steps 5000 --lr_scheduler_type linear --learning_rate ${learning_rate} --pad_to_max_length  --beta_schedule squaredcos_improved_ddpm --warmup_steps 2000 --logging_steps 50 --save_steps 1000    --load_states_in_eval_from_model_path true --max_eval_samples 96 ${PARAMS_FOR_LOCAL} --weight_decay 0.0 --save_checkpoints_on_s3 --dataset_folder "/net/nfs.cirrascale/s2-research/rabeehk/simplex-diffusion/datasets/qqp/"

#model_path="/net/nfs.cirrascale/s2-research/rabeehk/outputs/paper_experiments/qqp_self_condition_ablations/ours_lr_"${learning_rate}"_steps_"${max_steps}"_self_logits/checkpoint-40000/"
#python -m torch.distributed.launch --nproc_per_node 8 run_simplification.py --model_name_or_path ${model_path} --do_eval --do_predict --dataset_name qqp  --output_dir ${model_path}   --per_device_train_batch_size=1 --per_device_eval_batch_size=12   --report_to tensorboard --eval_steps 1000  --max_steps ${max_steps} --max_source_length 100  --max_target_length 85 --max_seq_length 185 --conditional_generation "seq2seq" --num_inference_diffusion_steps ${num_inference_diffusion_steps} --evaluation_strategy steps --simplex_value 5 --num_diffusion_steps 5000 --lr_scheduler_type linear --learning_rate ${learning_rate} --pad_to_max_length  --beta_schedule squaredcos_improved_ddpm --warmup_steps 2000 --logging_steps 50 --save_steps 1000    --self_condition "logits"  --load_states_in_eval_from_model_path true --max_eval_samples 96 ${PARAMS_FOR_LOCAL} --weight_decay 0.0 --save_checkpoints_on_s3 --dataset_folder "/net/nfs.cirrascale/s2-research/rabeehk/simplex-diffusion/datasets/qqp/"

#model_path="/net/nfs.cirrascale/s2-research/rabeehk/outputs/paper_experiments/qqp_self_condition_ablations/ours_lr_"${learning_rate}"_steps_"${max_steps}"_self_condition_logits_mean_mix_before_weights/checkpoint-40000"
#python -m torch.distributed.launch --nproc_per_node 8 run_simplification.py --model_name_or_path ${model_path}  --do_eval --do_predict --dataset_name qqp  --output_dir ${model_path}  --per_device_train_batch_size=1 --per_device_eval_batch_size=12   --report_to tensorboard --eval_steps 1000  --max_steps ${max_steps} --max_source_length 100  --max_target_length 85 --max_seq_length 185 --conditional_generation "seq2seq" --num_inference_diffusion_steps ${num_inference_diffusion_steps} --evaluation_strategy steps --simplex_value 5 --num_diffusion_steps 5000 --lr_scheduler_type linear --learning_rate ${learning_rate} --pad_to_max_length  --beta_schedule squaredcos_improved_ddpm --warmup_steps 2000 --logging_steps 50 --save_steps 1000    --self_condition "logits_mean"  --self_condition_mix_before_weights true --load_states_in_eval_from_model_path true --max_eval_samples 96 ${PARAMS_FOR_LOCAL} --weight_decay 0.0 --save_checkpoints_on_s3 --dataset_folder "/net/nfs.cirrascale/s2-research/rabeehk/simplex-diffusion/datasets/qqp/"


# debug
#model_path="/net/nfs.cirrascale/s2-research/rabeehk/outputs/paper_experiments/qqp_self_condition_ablations/ours_lr_"${learning_rate}"_steps_"${max_steps}"_self_condition_logits_mean_mix_before_weights/checkpoint-40000"
#python -m torch.distributed.launch --nproc_per_node 8 run_simplification.py --model_name_or_path ${model_path}  --do_eval --do_predict --dataset_name qqp  --output_dir "debug"  --per_device_train_batch_size=1 --per_device_eval_batch_size=12   --report_to tensorboard --eval_steps 1000  --max_steps ${max_steps} --max_source_length 100  --max_target_length 85 --max_seq_length 185 --conditional_generation "seq2seq" --num_inference_diffusion_steps 10 --evaluation_strategy steps --simplex_value 5 --num_diffusion_steps 5000 --lr_scheduler_type linear --learning_rate ${learning_rate} --pad_to_max_length  --beta_schedule squaredcos_improved_ddpm --warmup_steps 2000 --logging_steps 50 --save_steps 1000    --self_condition "logits_mean"  --self_condition_mix_before_weights true --load_states_in_eval_from_model_path true --max_eval_samples 96 ${PARAMS_FOR_LOCAL} --weight_decay 0.0 --save_checkpoints_on_s3 --dataset_folder "/net/nfs.cirrascale/s2-research/rabeehk/simplex-diffusion/datasets/qqp/" --max_predict_samples 10
