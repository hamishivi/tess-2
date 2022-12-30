# Running the UL2 method on the cloud. rabeehk-1

python -m torch.distributed.launch     --nproc_per_node 16  run_mlm.py     --model_name_or_path roberta-large     --per_device_train_batch_size 24     --per_device_eval_batch_size 6    --do_train     --do_eval     --output_dir /home/rabeehk/outputs/opentext_ul2_objective_lr_1e-4_length_256/ --evaluation_strategy steps --eval_steps 1000 --report_to  tensorboard --overwrite_output_dir  --max_seq_length 256  --max_eval_samples 96 --simplex_value 5  --num_diffusion_steps 5000  --num_inference_diffusion_steps 2500 --lr_scheduler_type cosine --learning_rate 1e-4 --pad_to_max_length  --beta_schedule squaredcos_improved_ddpm --weight_decay 0.01 --tokenized_data_path processed_data/openwebtext_256_split/  --top_p 0.99 --max_steps 2000000 --gradient_accumulation_steps 4 --warmup_steps 2000 --logging_steps 50   --save_steps 1000  --conditional_generation "ul2"


# Running the UL2 method with self-conditioning - we needed to reduce the batch_size.
python -m torch.distributed.launch     --nproc_per_node 16  run_mlm.py     --model_name_or_path roberta-large     --per_device_train_batch_size 12    --per_device_eval_batch_size 6    --do_train     --do_eval     --output_dir /home/rabeehk/outputs/opentext_ul2_objective_lr_1e-4_length_256_with_self_condition_logits_addition/ --evaluation_strategy steps --eval_steps 1000 --report_to  tensorboard --overwrite_output_dir  --max_seq_length 256  --max_eval_samples 96 --simplex_value 5  --num_diffusion_steps 5000  --num_inference_diffusion_steps 2500 --lr_scheduler_type cosine --learning_rate 1e-4 --pad_to_max_length  --beta_schedule squaredcos_improved_ddpm --weight_decay 0.01 --tokenized_data_path processed_data/openwebtext_256_split/  --top_p 0.99 --max_steps 2000000 --gradient_accumulation_steps 8 --warmup_steps 2000 --logging_steps 50   --save_steps 1000  --conditional_generation "ul2" --self_condition logits_addition


# Running the UL2 method with self-conditioning original setup - we needed to reduce the batch_size.
python -m torch.distributed.launch     --nproc_per_node 16  run_mlm.py     --model_name_or_path roberta-large     --per_device_train_batch_size 12    --per_device_eval_batch_size 6    --do_train     --do_eval     --output_dir /home/rabeehk/outputs/opentext_ul2_objective_lr_1e-4_length_256_with_self_condition_logits/ --evaluation_strategy steps --eval_steps 1000 --report_to  tensorboard --overwrite_output_dir  --max_seq_length 256  --max_eval_samples 96 --simplex_value 5  --num_diffusion_steps 5000  --num_inference_diffusion_steps 2500 --lr_scheduler_type cosine --learning_rate 1e-4 --pad_to_max_length  --beta_schedule squaredcos_improved_ddpm --weight_decay 0.01 --tokenized_data_path processed_data/openwebtext_256_split/  --top_p 0.99 --max_steps 2000000 --gradient_accumulation_steps 8 --warmup_steps 2000 --logging_steps 50   --save_steps 1000  --conditional_generation "ul2" --self_condition logits


# Running glue baseline
We use 5 epochs for wnli and mrpc, rest 3 epoch.


python run_glue.py --model_name_or_path roberta-large  --dataset_name wnli --do_train --do_eval --do_predict --max_seq_length 128 --per_device_train_batch_size 32 --per_device_eval_batch_size 32 --evaluation_strategy epoch --save_strategy epoch  --output_dir /net/nfs.cirrascale/s2-research/rabeehk/outputs/simplex_new/glue_roberta_large_baseline/wnli --report_to tensorboard  --overwrite_output_dir --pad_to_max_length --learning_rate 2e-5 --num_train_epochs 3 --logging_steps 50  --load_best_model_at_end --checkpoint_best_model --greater_is_better true 



python run_glue.py --model_name_or_path roberta-large  --dataset_name wnli --do_train --do_eval --do_predict --max_seq_length 128 --per_device_train_batch_size 100 --per_device_eval_batch_size 100 --evaluation_strategy epoch --save_strategy epoch  --output_dir /net/nfs.cirrascale/s2-research/rabeehk/outputs/simplex_new/glue_roberta_large_baseline/wnli --report_to tensorboard  --overwrite_output_dir --pad_to_max_length --learning_rate 2e-5 --num_train_epochs 3 --logging_steps 50  --load_best_model_at_end true --checkpoint_best_model --greater_is_better true --warmup_steps 500 --save_steps 1000 --tokenizer_name roberta-large --save_total_limit 1 

