# Train the models.

BASE_DIR="/net/nfs.cirrascale/s2-research/rabeehk/"
shared_params="--model_name_or_path roberta-large --per_device_train_batch_size 24 --per_device_eval_batch_size 6 --do_train --do_eval  --evaluation_strategy steps --eval_steps 1000 --report_to tensorboard --overwrite_output_dir --max_seq_length 256 --max_eval_samples 48 --simplex_value 5 --num_diffusion_steps 5000 --num_inference_diffusion_steps 1000 --lr_scheduler_type cosine --learning_rate 1e-4 --pad_to_max_length --beta_schedule squaredcos_improved_ddpm --weight_decay 0.01 --tokenized_data_path processed_data/openwebtext_256_split/ --top_p 0.99 --max_steps 100000 --gradient_accumulation_steps 16 --warmup_steps 2000 --logging_steps 50 --save_steps 1000 --conditional_generation ul2"
PARAMS_FOR_LOCAL=" --save_total_limit 1"


# Train the base model
python -m torch.distributed.launch --nproc_per_node 4 run_mlm.py --output_dir $BASE_DIR"/outputs/paper_experiments/tune_lr_scheduler/ul2_cosine_lr_scheduler" ${shared_params} ${PARAMS_FOR_LOCAL}



