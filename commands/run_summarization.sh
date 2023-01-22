DEBUG_params="--num_inference_diffusion_steps 10 --max_eval_samples 10 --per_device_train_batch_size 4 --per_device_eval_batch_size 4 --eval_steps 10"
PARAMS_FOR_LOCAL=" --save_total_limit 1 "

# Debug model.
#python run_summarization.py --model_name_or_path roberta-large --do_train --do_eval --dataset_name xsum --dataset_config "3.0.0" --output_dir /net/nfs.cirrascale/s2-research/rabeehk/outputs/debug/ --per_device_train_batch_size=12 --per_device_eval_batch_size=24 --overwrite_output_dir  --report_to tensorboard --eval_steps 1000  --max_steps 1000000 --max_eval_samples 96 --max_source_length 512 --max_target_length 128 --max_seq_length 640 --conditional_generation "seq2seq" --num_inference_diffusion_steps 2500 --evaluation_strategy steps --simplex_value 5 --num_diffusion_steps 5000 --lr_scheduler_type linear --learning_rate 1e-4 --pad_to_max_length false --beta_schedule squaredcos_improved_ddpm --weight_decay 0.01 --top_p 0.99 --warmup_steps 2000 --logging_steps 50 --save_steps 1000 ${DEBUG_params} --resize_position_embeddings True --resize_position_embeddings_alternatively False

# data length=512
# learning_rate=5e-5
#python -m torch.distributed.launch --nproc_per_node 4 run_summarization.py --model_name_or_path roberta-large --do_train --do_eval --dataset_name xsum --dataset_config "3.0.0" --output_dir "/net/nfs.cirrascale/s2-research/rabeehk/outputs/paper_experiments/summarization_results/ours_lr_"${learning_rate} --per_device_train_batch_size=6 --per_device_eval_batch_size=12 --overwrite_output_dir  --report_to tensorboard --eval_steps 1000  --max_steps 100000 --max_eval_samples 96 --max_source_length 512  --max_target_length 120 --max_seq_length 632 --conditional_generation "seq2seq" --num_inference_diffusion_steps 1000 --evaluation_strategy steps --simplex_value 5 --num_diffusion_steps 5000 --lr_scheduler_type linear --learning_rate ${learning_rate} --pad_to_max_length  --beta_schedule squaredcos_improved_ddpm --weight_decay 0.0 --top_p 0.99 --warmup_steps 2000 --logging_steps 50 --save_steps 1000 ${PARAMS_FOR_LOCAL} --resize_position_embeddings True  --self_condition "logits_mean" --self_condition_mix_before_weights true  --gradient_accumulation_steps 2  --save_checkpoints_on_s3


# with alternative resize => does not help too.
# learning_rate=2e-5
# python -m torch.distributed.launch --nproc_per_node 8 run_summarization.py --model_name_or_path roberta-large --do_train --do_eval --dataset_name xsum --dataset_config "3.0.0" --output_dir "/net/nfs.cirrascale/s2-research/rabeehk/outputs/paper_experiments/summarization_results/ours_lr_"${learning_rate}"_alternative_resize" --per_device_train_batch_size=6 --per_device_eval_batch_size=12 --overwrite_output_dir  --report_to tensorboard --eval_steps 1000  --max_steps 100000 --max_eval_samples 96 --max_source_length 512  --max_target_length 120 --max_seq_length 632 --conditional_generation "seq2seq" --num_inference_diffusion_steps 1000 --evaluation_strategy steps --simplex_value 5 --num_diffusion_steps 5000 --lr_scheduler_type linear --learning_rate ${learning_rate} --pad_to_max_length  --beta_schedule squaredcos_improved_ddpm --weight_decay 0.0 --top_p 0.99 --warmup_steps 2000 --logging_steps 50 --save_steps 1000 ${PARAMS_FOR_LOCAL} --resize_position_embeddings True  --self_condition "logits_mean" --self_condition_mix_before_weights true  --gradient_accumulation_steps 1  --save_checkpoints_on_s3 --resize_position_embeddings_alternatively True


# Without resize position embeddings.
learning_rate=2e-5
max_steps=60000
#python -m torch.distributed.launch --nproc_per_node 8 run_summarization.py --model_name_or_path roberta-large --do_train --do_eval --dataset_name xsum --dataset_config "3.0.0" --output_dir "/net/nfs.cirrascale/s2-research/rabeehk/outputs/paper_experiments/summarization_results/ours_lr_"${learning_rate}"_max_steps_"${max_steps} --per_device_train_batch_size=6 --per_device_eval_batch_size=12 --overwrite_output_dir  --report_to tensorboard --eval_steps 1000  --max_steps ${max_steps} --max_eval_samples 96 --max_source_length 392  --max_target_length 120 --max_seq_length 512  --conditional_generation "seq2seq" --num_inference_diffusion_steps 1000 --evaluation_strategy steps --simplex_value 5 --num_diffusion_steps 5000 --lr_scheduler_type linear --learning_rate ${learning_rate} --pad_to_max_length  --beta_schedule squaredcos_improved_ddpm --weight_decay 0.0 --top_p 0.99 --warmup_steps 2000 --logging_steps 50 --save_steps 1000 ${PARAMS_FOR_LOCAL} --self_condition "logits_mean" --self_condition_mix_before_weights true  --gradient_accumulation_steps 1  --save_checkpoints_on_s3

# same model on base size.
learning_rate=2e-5
max_steps=60000
model_name="roberta-base"
#python -m torch.distributed.launch --nproc_per_node 8 run_summarization.py --model_name_or_path ${model_name} --do_train --do_eval --dataset_name xsum --dataset_config "3.0.0" --output_dir "/net/nfs.cirrascale/s2-research/rabeehk/outputs/paper_experiments/summarization_results/ours_lr_"${learning_rate}"_max_steps_"${max_steps}"_model_"${model_name} --per_device_train_batch_size=6 --per_device_eval_batch_size=12 --overwrite_output_dir  --report_to tensorboard --eval_steps 1000  --max_steps ${max_steps} --max_eval_samples 96 --max_source_length 392  --max_target_length 120 --max_seq_length 512  --conditional_generation "seq2seq" --num_inference_diffusion_steps 1000 --evaluation_strategy steps --simplex_value 5 --num_diffusion_steps 5000 --lr_scheduler_type linear --learning_rate ${learning_rate} --pad_to_max_length  --beta_schedule squaredcos_improved_ddpm --weight_decay 0.0 --top_p 0.99 --warmup_steps 2000 --logging_steps 50 --save_steps 1000 ${PARAMS_FOR_LOCAL} --self_condition "logits_mean" --self_condition_mix_before_weights true  --gradient_accumulation_steps 1  --save_checkpoints_on_s3



# Evaluate the trained models.
learning_rate=2e-5
max_steps=60000
checkpoint_id=50000
checkpoint="/net/nfs.cirrascale/s2-research/rabeehk/outputs/paper_experiments/summarization_results/ours_lr_"${learning_rate}"_max_steps_"${max_steps}"/checkpoint-"${checkpoint_id}
python -m torch.distributed.launch --nproc_per_node 8 run_summarization.py --model_name_or_path ${checkpoint} --do_predict --dataset_name xsum --dataset_config "3.0.0" --output_dir ${checkpoint} --per_device_train_batch_size=6 --per_device_eval_batch_size=12 --overwrite_output_dir  --report_to tensorboard --eval_steps 1000  --max_steps ${max_steps} --max_eval_samples 96 --max_source_length 392  --max_target_length 120 --max_seq_length 512  --conditional_generation "seq2seq" --num_inference_diffusion_steps 1000 --evaluation_strategy steps --simplex_value 5 --num_diffusion_steps 5000 --lr_scheduler_type linear --learning_rate ${learning_rate} --pad_to_max_length  --beta_schedule squaredcos_improved_ddpm --weight_decay 0.0 --top_p 0.99 --warmup_steps 2000 --logging_steps 50 --save_steps 1000 ${PARAMS_FOR_LOCAL} --self_condition "logits_mean" --self_condition_mix_before_weights true  --gradient_accumulation_steps 1  --save_checkpoints_on_s3 --model_name_or_path ${checkpoint} --load_states_in_eval_from_model_path 




: '
# data length=1024
python run_summarization.py --model_name_or_path roberta-large --do_train --do_eval --dataset_name xsum --dataset_config "3.0.0" --output_dir /net/nfs.cirrascale/s2-research/rabeehk/outputs/paper_experiments/summarization_1024 --per_device_train_batch_size=6 --per_device_eval_batch_size=10 --overwrite_output_dir  --report_to tensorboard --eval_steps 1000  --max_steps 1000000 --max_eval_samples 96 --max_source_length 1024  --max_target_length 120 --max_seq_length 1144 --conditional_generation "seq2seq" --num_inference_diffusion_steps 1000 --evaluation_strategy steps --simplex_value 5 --num_diffusion_steps 5000 --lr_scheduler_type linear --learning_rate 1e-4 --pad_to_max_length --beta_schedule squaredcos_improved_ddpm --weight_decay 0.01 --top_p 0.99 --warmup_steps 2000 --logging_steps 50 --save_steps 1000 ${PARAMS_FOR_LOCAL} --resize_position_embeddings True  --gradient_accumulation_steps 2 --fp16 --self_condition "logits_mean"
'
