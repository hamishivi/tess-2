PARAMS_FOR_LOCAL=" --save_total_limit 1 "

# Run simplification.
learning_rate=5e-5
# python run_simplification.py --model_name_or_path facebook/bart-large --do_train --do_eval --dataset_name wikilarge  --output_dir "/net/nfs.cirrascale/s2-research/rabeehk/outputs/paper_experiments/tune_lrs_simplification/lr_"${learning_rate}"_simplification_baseline" --per_device_train_batch_size=12 --per_device_eval_batch_size=25 --overwrite_output_dir  --report_to tensorboard --eval_steps 1000  --max_steps 500000 --max_eval_samples 96 --max_source_length 64  --max_target_length 64  --evaluation_strategy steps  --lr_scheduler_type linear --learning_rate ${learning_rate} --pad_to_max_length  --warmup_steps 2000 --logging_steps 50 --save_steps 1000 ${PARAMS_FOR_LOCAL}   --predict_with_generate 

python run_simplification.py --model_name_or_path facebook/bart-large --do_train --do_eval --dataset_name wikilarge  --output_dir "/net/nfs.cirrascale/s2-research/rabeehk/outputs/paper_experiments/tune_lrs_simplification/lr_"${learning_rate}"_simplification_baseline_with_wd" --per_device_train_batch_size=12 --per_device_eval_batch_size=25 --overwrite_output_dir  --report_to tensorboard --eval_steps 1000  --max_steps 500000 --max_eval_samples 96 --max_source_length 64  --max_target_length 64  --evaluation_strategy steps  --lr_scheduler_type linear --learning_rate ${learning_rate} --pad_to_max_length  --warmup_steps 2000 --logging_steps 50 --save_steps 1000 ${PARAMS_FOR_LOCAL}   --predict_with_generate  --weight_decay 0.01 


# Run summarization.
# data length=512
: '
python -m torch.distributed.launch --nproc_per_node 4 run_summarization.py --model_name_or_path facebook/bart-large --do_train --do_eval --dataset_name xsum --dataset_config "3.0.0" --output_dir /net/nfs.cirrascale/s2-research/rabeehk/outputs/paper_experiments/summarization_512_baseline_bart_large --per_device_train_batch_size=6 --per_device_eval_batch_size=15 --overwrite_output_dir  --report_to tensorboard --eval_steps 1000  --max_steps 1000000 --max_eval_samples 96 --max_source_length 512  --max_target_length 120   --evaluation_strategy steps  --lr_scheduler_type linear --learning_rate 1e-4 --pad_to_max_length  --weight_decay 0.01 --warmup_steps 2000 --logging_steps 50 --save_steps 1000 ${PARAMS_FOR_LOCAL} --resize_position_embeddings True --fp16 --gradient_accumulation_steps 2  --predict_with_generate
'


: '
# Debug
python run_summarization.py --model_name_or_path facebook/bart-large --do_train --do_eval --dataset_name xsum --dataset_config "3.0.0" --output_dir debug  --per_device_train_batch_size=6 --per_device_eval_batch_size=15 --overwrite_output_dir  --report_to tensorboard --eval_steps 1000  --max_steps 1000000 --max_eval_samples 96 --max_source_length 512  --max_target_length 120   --evaluation_strategy steps  --lr_scheduler_type linear --learning_rate 1e-4 --pad_to_max_length  --weight_decay 0.01 --warmup_steps 2000 --logging_steps 50 --save_steps 1000 ${PARAMS_FOR_LOCAL} --resize_position_embeddings True --fp16 --gradient_accumulation_steps 2  --predict_with_generate
'

