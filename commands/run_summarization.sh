DEBUG_params="--num_inference_diffusion_steps 10 --max_eval_samples 10 --per_device_train_batch_size 4 --per_device_eval_batch_size 4 --eval_steps 10"
PARAMS_FOR_LOCAL=" --save_total_limit 1 "

: '
# Debug model.
python run_summarization.py --model_name_or_path roberta-large --do_train --do_eval --dataset_name xsum --dataset_config "3.0.0" --output_dir /net/nfs.cirrascale/s2-research/rabeehk/outputs/simplex_new/summarization_512 --per_device_train_batch_size=12 --per_device_eval_batch_size=24 --overwrite_output_dir  --report_to tensorboard --eval_steps 1000  --max_steps 1000000 --max_eval_samples 96 --max_source_length 512 --max_target_length 128 --max_seq_length 640 --conditional_generation "seq2seq" --num_inference_diffusion_steps 2500 --evaluation_strategy steps --simplex_value 5 --num_diffusion_steps 5000 --lr_scheduler_type cosine --learning_rate 1e-4 --pad_to_max_length false --beta_schedule squaredcos_improved_ddpm --weight_decay 0.01 --top_p 0.99 --warmup_steps 2000 --logging_steps 50 --save_steps 1000 ${DEBUG_params} --resize_position_embeddings True
'

: '
# data length=512
python -m torch.distributed.launch --nproc_per_node 4 run_summarization.py --model_name_or_path roberta-large --do_train --do_eval --dataset_name xsum --dataset_config "3.0.0" --output_dir /net/nfs.cirrascale/s2-research/rabeehk/outputs/paper_experiments/summarization_512 --per_device_train_batch_size=12 --per_device_eval_batch_size=15 --overwrite_output_dir  --report_to tensorboard --eval_steps 1000  --max_steps 1000000 --max_eval_samples 96 --max_source_length 512  --max_target_length 120 --max_seq_length 632 --conditional_generation "seq2seq" --num_inference_diffusion_steps 1000 --evaluation_strategy steps --simplex_value 5 --num_diffusion_steps 5000 --lr_scheduler_type cosine --learning_rate 1e-4 --pad_to_max_length  --beta_schedule squaredcos_improved_ddpm --weight_decay 0.01 --top_p 0.99 --warmup_steps 2000 --logging_steps 50 --save_steps 1000 ${PARAMS_FOR_LOCAL} --resize_position_embeddings True --fp16
'

# data length=1024
python run_summarization.py --model_name_or_path roberta-large --do_train --do_eval --dataset_name xsum --dataset_config "3.0.0" --output_dir /net/nfs.cirrascale/s2-research/rabeehk/outputs/paper_experiments/summarization_1024 --per_device_train_batch_size=6 --per_device_eval_batch_size=10 --overwrite_output_dir  --report_to tensorboard --eval_steps 1000  --max_steps 1000000 --max_eval_samples 96 --max_source_length 1024  --max_target_length 120 --max_seq_length 1144 --conditional_generation "seq2seq" --num_inference_diffusion_steps 1000 --evaluation_strategy steps --simplex_value 5 --num_diffusion_steps 5000 --lr_scheduler_type cosine --learning_rate 1e-4 --pad_to_max_length --beta_schedule squaredcos_improved_ddpm --weight_decay 0.01 --top_p 0.99 --warmup_steps 2000 --logging_steps 50 --save_steps 1000 ${PARAMS_FOR_LOCAL} --resize_position_embeddings True  --gradient_accumulation_steps 2 --fp16
