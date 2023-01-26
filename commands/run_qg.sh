# trains on the qg datasets for paraphrase generation.
PARAMS_FOR_LOCAL=" --save_total_limit 1 "

learning_rate=3e-5
num_inference_diffusion_steps=1000
max_steps=120000 #80000 # We try for 80000 
dataset="qg"
python -m torch.distributed.launch --nproc_per_node 8 run_simplification.py --model_name_or_path "roberta-base" --do_train --do_eval --do_predict --dataset_name ${dataset}  --output_dir  "/net/nfs.cirrascale/s2-research/rabeehk/outputs/paper_experiments/"${dataset}"_tune_steps/ours_lr_"${learning_rate}"_steps_"${max_steps} --per_device_train_batch_size=1 --per_device_eval_batch_size=12   --report_to tensorboard --eval_steps 1000  --max_steps ${max_steps} --max_source_length 155  --max_target_length 65 --max_seq_length 220 --conditional_generation "seq2seq" --num_inference_diffusion_steps ${num_inference_diffusion_steps} --evaluation_strategy steps --simplex_value 5 --num_diffusion_steps 5000 --lr_scheduler_type linear --learning_rate ${learning_rate} --pad_to_max_length  --beta_schedule squaredcos_improved_ddpm  --warmup_steps 2000 --logging_steps 50 --save_steps 1000    --self_condition "logits_mean"  --self_condition_mix_before_weights true --load_states_in_eval_from_model_path true --max_eval_samples 96 ${PARAMS_FOR_LOCAL} --weight_decay 0.0 --save_checkpoints_on_s3 --dataset_folder "/net/nfs.cirrascale/s2-research/rabeehk/simplex-diffusion/datasets/"${dataset}"/"

# **** reported for steps=120k, lr=3e-5 ****
: '
for max_steps in  40000 80000 
do
model_path="/net/nfs.cirrascale/s2-research/rabeehk/outputs/paper_experiments/"${dataset}"_tune_steps/ours_lr_"${learning_rate}"_steps_"${max_steps}"/checkpoint-"${max_steps}
python -m torch.distributed.launch --nproc_per_node 8 run_simplification.py --model_name_or_path ${model_path}  --do_eval --do_predict --dataset_name ${dataset}  --output_dir  ${model_path} --per_device_train_batch_size=1 --per_device_eval_batch_size=12   --report_to tensorboard --eval_steps 1000  --max_steps ${max_steps} --max_source_length 155  --max_target_length 65 --max_seq_length 220 --conditional_generation "seq2seq" --num_inference_diffusion_steps ${num_inference_diffusion_steps} --evaluation_strategy steps --simplex_value 5 --num_diffusion_steps 5000 --lr_scheduler_type linear --learning_rate ${learning_rate} --pad_to_max_length  --beta_schedule squaredcos_improved_ddpm  --warmup_steps 2000 --logging_steps 50 --save_steps 1000    --self_condition "logits_mean"  --self_condition_mix_before_weights true --load_states_in_eval_from_model_path true --max_eval_samples 96 ${PARAMS_FOR_LOCAL} --weight_decay 0.0 --save_checkpoints_on_s3 --dataset_folder "/net/nfs.cirrascale/s2-research/rabeehk/simplex-diffusion/datasets/"${dataset}"/" --load_states_in_eval_from_model_path
done
'
