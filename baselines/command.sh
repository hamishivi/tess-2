PARAMS_FOR_LOCAL=" --save_total_limit 1 "

#######################
# Run simplification.
#######################
# not used.
#learning_rate=3e-5
#python run_simplification.py --model_name_or_path facebook/bart-large --do_train --do_eval --dataset_name wikilarge  --output_dir "/net/nfs.cirrascale/s2-research/rabeehk/outputs/paper_experiments/tune_lrs_simplification/lr_"${learning_rate}"_simplification_baseline" --per_device_train_batch_size=12 --per_device_eval_batch_size=25 --overwrite_output_dir  --report_to tensorboard --eval_steps 1000  --max_steps 500000 --max_eval_samples 96 --max_source_length 64  --max_target_length 64  --evaluation_strategy steps  --lr_scheduler_type linear --learning_rate ${learning_rate} --pad_to_max_length  --warmup_steps 2000 --logging_steps 50 --save_steps 1000 --predict_with_generate --resume_from_checkpoint "/net/nfs.cirrascale/s2-research/rabeehk/outputs/paper_experiments/tune_lrs_simplification/lr_3e-5_simplification_baseline/checkpoint-246000/" ${PARAMS_FOR_LOCAL}
# learning_rate=3e-5
# not used.
# python run_simplification.py --model_name_or_path facebook/bart-large --do_train --do_eval --dataset_name wikilarge  --output_dir "/net/nfs.cirrascale/s2-research/rabeehk/outputs/paper_experiments/tune_lrs_simplification/lr_"${learning_rate}"_simplification_baseline_with_wd" --per_device_train_batch_size=12 --per_device_eval_batch_size=25 --overwrite_output_dir  --report_to tensorboard --eval_steps 1000  --max_steps 500000 --max_eval_samples 96 --max_source_length 64  --max_target_length 64  --evaluation_strategy steps  --lr_scheduler_type linear --learning_rate ${learning_rate} --pad_to_max_length  --warmup_steps 2000 --logging_steps 50 --save_steps 1000    --predict_with_generate  --weight_decay 0.01 --resume_from_checkpoint "/net/nfs.cirrascale/s2-research/rabeehk/outputs/paper_experiments/tune_lrs_simplification/lr_3e-5_simplification_baseline_with_wd/checkpoint-245000/" ${PARAMS_FOR_LOCAL}
# lr=3e-5 is the best and also without wd is the best.

# *** Running the baseline ***
# learning_rate=2e-5
# python -m torch.distributed.launch --nproc_per_node 4 run_simplification.py --model_name_or_path facebook/bart-large --do_train --do_eval --dataset_name wikilarge  --output_dir "/net/nfs.cirrascale/s2-research/rabeehk/outputs/paper_experiments/simplification_results/baseline_lr_"${learning_rate}"_no_wd" --per_device_train_batch_size=1 --per_device_eval_batch_size=12 --overwrite_output_dir  --report_to tensorboard --eval_steps 1000  --max_steps 500000 --max_eval_samples 96 --max_source_length 64  --max_target_length 64  --evaluation_strategy steps  --lr_scheduler_type linear --learning_rate ${learning_rate} --pad_to_max_length  --warmup_steps 2000 --logging_steps 50 --save_steps 1000 --predict_with_generate  ${PARAMS_FOR_LOCAL} --save_checkpoints_on_s3 --gradient_accumulation_steps 2 


# Run summarization.
# data length=512
learning_rate=5e-5
max_steps=60000
# max position embedding for BART is 1024, so we do not need to resize it.
#python -m torch.distributed.launch --nproc_per_node 8 run_summarization.py --model_name_or_path facebook/bart-large --do_train --do_eval --dataset_name xsum --dataset_config "3.0.0" --output_dir "/net/nfs.cirrascale/s2-research/rabeehk/outputs/paper_experiments/summarization_results/baseline_lr_"${learning_rate}"_steps_"${max_steps} --per_device_train_batch_size=6 --per_device_eval_batch_size=12 --overwrite_output_dir  --report_to tensorboard --eval_steps 1000  --max_steps ${max_steps} --max_eval_samples 96 --max_source_length 392  --max_target_length 120   --evaluation_strategy steps  --lr_scheduler_type linear --learning_rate ${learning_rate} --pad_to_max_length  --weight_decay 0.0 --warmup_steps 2000 --logging_steps 50 --save_steps 1000 ${PARAMS_FOR_LOCAL}  --gradient_accumulation_steps 1  --predict_with_generate --save_checkpoints_on_s3 


# Evaluate the summarizaiton.
learning_rate=2e-5
max_steps=60000
# max position embedding for BART is 1024, so we do not need to resize it.
for learning_rate in 2e-5 3e-5
do
   for checkpoint in 50000 55000
   do 	   
   checkpoint="/net/nfs.cirrascale/s2-research/rabeehk/outputs/paper_experiments/summarization_results/baseline_lr_"${learning_rate}"_steps_"${max_steps}"/checkpoint-"${checkpoint}
   python -m torch.distributed.launch --nproc_per_node 8 run_summarization.py --model_name_or_path ${checkpoint} --do_predict --dataset_name xsum --dataset_config "3.0.0" --output_dir "/net/nfs.cirrascale/s2-research/rabeehk/outputs/paper_experiments/summarization_results/baseline_lr_"${learning_rate}"_steps_"${max_steps} --per_device_train_batch_size=6 --per_device_eval_batch_size=12 --overwrite_output_dir  --report_to tensorboard --eval_steps 1000  --max_steps ${max_steps} --max_eval_samples 96 --max_source_length 392  --max_target_length 120   --evaluation_strategy steps  --lr_scheduler_type linear --learning_rate ${learning_rate} --pad_to_max_length  --weight_decay 0.0 --warmup_steps 2000 --logging_steps 50 --save_steps 1000 ${PARAMS_FOR_LOCAL}  --gradient_accumulation_steps 1  --predict_with_generate --save_checkpoints_on_s3 --load_states_in_eval_from_model_path true 
   done
done



#####################################################################
# *** Running glue baseline ****
#DATASET="sst2"
#python run_glue.py --model_name_or_path roberta-large  --dataset_name ${DATASET} --do_train --do_eval --do_predict --max_seq_length 128 --per_device_train_batch_size 64 --per_device_eval_batch_size 64 --evaluation_strategy epoch --save_strategy epoch  --output_dir "/net/nfs.cirrascale/s2-research/rabeehk/outputs/paper_experiments/glue_results/baseline_"${DATASET} --report_to tensorboard  --overwrite_output_dir --pad_to_max_length --learning_rate 3e-5 --num_train_epochs 3 --logging_steps 50  --load_best_model_at_end true --checkpoint_best_model --greater_is_better true --warmup_steps 500  --tokenizer_name roberta-large --save_total_limit 1 --lr_scheduler_type linear  --gradient_accumulation_steps 2

# 10 epochs for mrpc and rte, wnli,stsb cola, 3 for the rest.
# DATASET="stsb"
#python run_glue.py --model_name_or_path roberta-large  --dataset_name ${DATASET} --do_train --do_eval --do_predict --max_seq_length 128 --per_device_train_batch_size 64 --per_device_eval_batch_size 64 --evaluation_strategy epoch --save_strategy epoch  --output_dir "/net/nfs.cirrascale/s2-research/rabeehk/outputs/paper_experiments/glue_results/baseline_"${DATASET} --report_to tensorboard  --overwrite_output_dir --pad_to_max_length --learning_rate 3e-5 --num_train_epochs 10 --logging_steps 50  --load_best_model_at_end true --checkpoint_best_model --greater_is_better true --warmup_steps 500  --tokenizer_name roberta-large --save_total_limit 1 --lr_scheduler_type linear  --gradient_accumulation_steps 2

#####################################################################

# DEBUG
# python run_simplification.py --model_name_or_path facebook/bart-large --do_train --do_eval --dataset_name wikilarge  --output_dir "/net/nfs.cirrascale/s2-research/rabeehk/outputs/paper_experiments/debug/" --per_device_train_batch_size=12 --per_device_eval_batch_size=25 --overwrite_output_dir  --report_to tensorboard --eval_steps 1000  --max_steps 500000 --num_train_epochs 5 --max_source_length 64  --max_target_length 64  --evaluation_strategy steps  --lr_scheduler_type linear --learning_rate ${learning_rate} --pad_to_max_length  --warmup_steps 2000 --logging_steps 50 --save_steps 1000 ${PARAMS_FOR_LOCAL}   --predict_with_generate  --weight_decay 0.01 

: '
# Debug
python run_summarization.py --model_name_or_path facebook/bart-large --do_train --do_eval --dataset_name xsum --dataset_config "3.0.0" --output_dir debug  --per_device_train_batch_size=6 --per_device_eval_batch_size=15 --overwrite_output_dir  --report_to tensorboard --eval_steps 1000  --max_steps 1000000 --max_eval_samples 96 --max_source_length 512  --max_target_length 120   --evaluation_strategy steps  --lr_scheduler_type linear --learning_rate 1e-4 --pad_to_max_length  --weight_decay 0.01 --warmup_steps 2000 --logging_steps 50 --save_steps 1000 ${PARAMS_FOR_LOCAL} --resize_position_embeddings True --fp16 --gradient_accumulation_steps 2  --predict_with_generate
'
# DEBUG
# DATASET="mnli"
# python run_glue.py --model_name_or_path roberta-large  --dataset_name ${DATASET} --do_train --do_eval --do_predict --max_seq_length 128 --per_device_train_batch_size 64 --per_device_eval_batch_size 64 --evaluation_strategy epoch --save_strategy epoch  --output_dir "/net/nfs.cirrascale/s2-research/rabeehk/outputs/paper_experiments/debug" --report_to tensorboard  --overwrite_output_dir --pad_to_max_length --learning_rate 3e-5 --num_train_epochs 3 --logging_steps 50  --load_best_model_at_end true --checkpoint_best_model --greater_is_better true --warmup_steps 500  --tokenizer_name roberta-large --save_total_limit 1 --lr_scheduler_type linear  --gradient_accumulation_steps 2

