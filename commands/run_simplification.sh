DEBUG_params="--num_inference_diffusion_steps 10 --max_eval_samples 10 --per_device_train_batch_size 4 --per_device_eval_batch_size 4 --eval_steps 10"
PARAMS_FOR_LOCAL=" --save_total_limit 1 "


: '
# debug
python run_simplification.py --model_name_or_path roberta-large --do_train --do_eval --dataset_name asset  --output_dir /net/nfs.cirrascale/s2-research/rabeehk/outputs/paper_experiments/debug --per_device_train_batch_size=12 --per_device_eval_batch_size=64 --overwrite_output_dir  --report_to tensorboard --eval_steps 1000  --max_steps 1000000 --max_source_length 64  --max_target_length 64 --max_seq_length 128 --conditional_generation "seq2seq" --num_inference_diffusion_steps 200 --evaluation_strategy steps --simplex_value 5 --num_diffusion_steps 5000 --lr_scheduler_type linear --learning_rate 1e-4 --pad_to_max_length  --beta_schedule squaredcos_improved_ddpm --weight_decay 0.01 --top_p 0.99 --warmup_steps 2000 --logging_steps 50 --save_steps 1000 ${PARAMS_FOR_LOCAL}  --fp16 --self_condition "logits_mean" --eval_steps 10 --max_eval_samples 64
'

# data length=512
: '
num_inference_diffusion_steps=600
python run_simplification.py --model_name_or_path roberta-large --do_train --do_eval --dataset_name asset  --output_dir /net/nfs.cirrascale/s2-research/rabeehk/outputs/paper_experiments/simplification_${num_inference_diffusion_steps} --per_device_train_batch_size=12 --per_device_eval_batch_size=15 --overwrite_output_dir  --report_to tensorboard --eval_steps 1000  --max_steps 1000000 --max_eval_samples 96 --max_source_length 64  --max_target_length 64 --max_seq_length 128 --conditional_generation "seq2seq" --num_inference_diffusion_steps ${num_inference_diffusion_steps} --evaluation_strategy steps --simplex_value 5 --num_diffusion_steps 5000 --lr_scheduler_type linear --learning_rate 1e-4 --pad_to_max_length  --beta_schedule squaredcos_improved_ddpm --weight_decay 0.01 --top_p 0.99 --warmup_steps 2000 --logging_steps 50 --save_steps 1000 ${PARAMS_FOR_LOCAL}  --fp16 --self_condition "logits_mean"
'

# Running on all the data.
num_inference_diffusion_steps=200
python run_simplification.py --model_name_or_path roberta-large --do_train --do_eval --dataset_name asset  --output_dir /net/nfs.cirrascale/s2-research/rabeehk/outputs/paper_experiments/simplification_${num_inference_diffusion_steps}_all_data --per_device_train_batch_size=12 --per_device_eval_batch_size=15 --overwrite_output_dir  --report_to tensorboard --eval_steps 1000  --max_steps 1000000 --max_source_length 64  --max_target_length 64 --max_seq_length 128 --conditional_generation "seq2seq" --num_inference_diffusion_steps ${num_inference_diffusion_steps} --evaluation_strategy steps --simplex_value 5 --num_diffusion_steps 5000 --lr_scheduler_type linear --learning_rate 1e-4 --pad_to_max_length  --beta_schedule squaredcos_improved_ddpm --weight_decay 0.01 --top_p 0.99 --warmup_steps 2000 --logging_steps 50 --save_steps 1000 ${PARAMS_FOR_LOCAL}  --fp16 --self_condition "logits_mean"



