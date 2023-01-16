PARAMS_FOR_LOCAL=" --save_total_limit 1 "


: '
python run_simplification.py \
    --model_name_or_path facebook/bart-base \
    --do_train \
    --do_eval \
    --dataset_name wikilarge \
    --source_prefix "summarize: " \
    --output_dir /tmp/tst-summarization \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=4 \
    --overwrite_output_dir \
    --predict_with_generate \
    --eval_steps 10 \
    --evaluation_strategy steps
'


python run_simplification.py --model_name_or_path facebook/bart-base --do_train --do_eval --dataset_name wikilarge  --output_dir /net/nfs.cirrascale/s2-research/rabeehk/outputs/paper_experiments/simplification_baseline --per_device_train_batch_size=12 --per_device_eval_batch_size=15 --overwrite_output_dir  --report_to tensorboard --eval_steps 1000  --max_steps 1000000 --max_eval_samples 96 --max_source_length 64  --max_target_length 64  --evaluation_strategy steps  --lr_scheduler_type cosine --learning_rate 1e-4 --pad_to_max_length   --weight_decay 0.01 --warmup_steps 2000 --logging_steps 50 --save_steps 1000 ${PARAMS_FOR_LOCAL}  --fp16 
~
~
