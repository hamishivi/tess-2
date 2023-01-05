# Running the UL2 method on the cloud. rabeehk-1

python -m torch.distributed.launch     --nproc_per_node 16  run_mlm.py     --model_name_or_path roberta-large     --per_device_train_batch_size 24     --per_device_eval_batch_size 6    --do_train     --do_eval     --output_dir /home/rabeehk/outputs/opentext_ul2_objective_lr_1e-4_length_256/ --evaluation_strategy steps --eval_steps 1000 --report_to  tensorboard --overwrite_output_dir  --max_seq_length 256  --max_eval_samples 96 --simplex_value 5  --num_diffusion_steps 5000  --num_inference_diffusion_steps 2500 --lr_scheduler_type cosine --learning_rate 1e-4 --pad_to_max_length  --beta_schedule squaredcos_improved_ddpm --weight_decay 0.01 --tokenized_data_path processed_data/openwebtext_256_split/  --top_p 0.99 --max_steps 2000000 --gradient_accumulation_steps 4 --warmup_steps 2000 --logging_steps 50   --save_steps 1000  --conditional_generation "ul2"


# Running the UL2 method with self-conditioning - we needed to reduce the batch_size. allennlp - rabeehk-3
python -m torch.distributed.launch     --nproc_per_node 16  run_mlm.py     --model_name_or_path roberta-large     --per_device_train_batch_size 12    --per_device_eval_batch_size 6    --do_train     --do_eval     --output_dir /home/rabeehk/outputs/opentext_ul2_objective_lr_1e-4_length_256_with_self_condition_logits_addition/ --evaluation_strategy steps --eval_steps 1000 --report_to  tensorboard --overwrite_output_dir  --max_seq_length 256  --max_eval_samples 96 --simplex_value 5  --num_diffusion_steps 5000  --num_inference_diffusion_steps 2500 --lr_scheduler_type cosine --learning_rate 1e-4 --pad_to_max_length  --beta_schedule squaredcos_improved_ddpm --weight_decay 0.01 --tokenized_data_path processed_data/openwebtext_256_split/  --top_p 0.99 --max_steps 2000000 --gradient_accumulation_steps 8 --warmup_steps 2000 --logging_steps 50   --save_steps 1000  --conditional_generation "ul2" --self_condition logits_addition


# Running the UL2 method with self-conditioning original setup - we needed to reduce the batch_size. x not yet.
python -m torch.distributed.launch     --nproc_per_node 16  run_mlm.py     --model_name_or_path roberta-large     --per_device_train_batch_size 12    --per_device_eval_batch_size 6    --do_train     --do_eval     --output_dir /home/rabeehk/outputs/opentext_ul2_objective_lr_1e-4_length_256_with_self_condition_logits/ --evaluation_strategy steps --eval_steps 1000 --report_to  tensorboard --overwrite_output_dir  --max_seq_length 256  --max_eval_samples 96 --simplex_value 5  --num_diffusion_steps 5000  --num_inference_diffusion_steps 2500 --lr_scheduler_type cosine --learning_rate 1e-4 --pad_to_max_length  --beta_schedule squaredcos_improved_ddpm --weight_decay 0.01 --tokenized_data_path processed_data/openwebtext_256_split/  --top_p 0.99 --max_steps 2000000 --gradient_accumulation_steps 8 --warmup_steps 2000 --logging_steps 50   --save_steps 1000  --conditional_generation "ul2" --self_condition logits


# Running the UL2 method with self-conditioning original setup - locally
python -m torch.distributed.launch     --nproc_per_node 8  run_mlm.py     --model_name_or_path roberta-large     --per_device_train_batch_size 12    --per_device_eval_batch_size 12    --do_train     --do_eval     --output_dir /net/nfs.cirrascale/s2-research/rabeehk/outputs/paper_experiments/opentext_ul2_objective_lr_1e-4_length_256_with_self_condition_logits/ --evaluation_strategy steps --eval_steps 1000 --report_to  tensorboard --overwrite_output_dir  --max_seq_length 256  --max_eval_samples 96 --simplex_value 5  --num_diffusion_steps 5000  --num_inference_diffusion_steps 2500 --lr_scheduler_type cosine --learning_rate 1e-4 --pad_to_max_length  --beta_schedule squaredcos_improved_ddpm --weight_decay 0.01 --tokenized_data_path processed_data/openwebtext_256_split/  --top_p 0.99 --max_steps 2000000 --gradient_accumulation_steps 16 --warmup_steps 2000 --logging_steps 50   --save_steps 1000  --conditional_generation "ul2" --self_condition logits


# Running  UL2 with self-conditioning with addition and classifier-free guidance - rabeehk-1 - S2
python -m torch.distributed.launch     --nproc_per_node 16  run_mlm.py     --model_name_or_path roberta-large     --per_device_train_batch_size 12    --per_device_eval_batch_size 6    --do_train     --do_eval     --output_dir /home/rabeehk/outputs/opentext_ul2_objective_lr_1e-4_length_256_with_self_condition_logits_addition_guidance_5/ --evaluation_strategy steps --eval_steps 1000 --report_to  tensorboard --overwrite_output_dir  --max_seq_length 256  --max_eval_samples 96 --simplex_value 5  --num_diffusion_steps 5000  --num_inference_diffusion_steps 2500 --lr_scheduler_type cosine --learning_rate 1e-4 --pad_to_max_length  --beta_schedule squaredcos_improved_ddpm --weight_decay 0.01 --tokenized_data_path processed_data/openwebtext_256_split/  --top_p 0.99 --max_steps 2000000 --gradient_accumulation_steps 8 --warmup_steps 2000 --logging_steps 50   --save_steps 1000  --conditional_generation "ul2" --self_condition logits_addition --guidance_scale 5







# Running glue baseline
# 5 epochs ( wnli, stsb), 10 epochs for mrpc and rte cola, 3 for the rest.
# no change for cola, wnli, stsb.
python run_glue.py --model_name_or_path roberta-large  --dataset_name wnli --do_train --do_eval --do_predict --max_seq_length 128 --per_device_train_batch_size 64 --per_device_eval_batch_size 64 --evaluation_strategy epoch --save_strategy epoch  --output_dir /net/nfs.cirrascale/s2-research/rabeehk/outputs/simplex_new/glue_roberta_large_baseline_tuned/wnli --report_to tensorboard  --overwrite_output_dir --pad_to_max_length --learning_rate 3e-5 --num_train_epochs 5 --logging_steps 50  --load_best_model_at_end true --checkpoint_best_model --greater_is_better true --warmup_steps 500  --tokenizer_name roberta-large --save_total_limit 1 --lr_scheduler_type cosine  --gradient_accumulation_steps 2




# our model on glue
# learning rate = 3e-5 is the best, 1e-4 is bad.
python run_glue.py --model_name_or_path roberta-large --dataset_name mrpc --output_dir /net/nfs.cirrascale/s2-research/rabeehk/outputs/simplex_new/ours_glue/lr_3e-5_mrpc --do_train --do_eval --do_predict --max_seq_length 128 --per_device_train_batch_size 64 --per_device_eval_batch_size 64 --evaluation_strategy steps --save_strategy steps   --report_to tensorboard --overwrite_output_dir --pad_to_max_length  --simplex_value 5 --num_diffusion_steps 5000 --num_inference_diffusion_steps 1000 --conditional_generation seq2seq  --learning_rate 3e-5 --gradient_accumulation_steps 2 --lr_scheduler_type cosine --beta_schedule squaredcos_improved_ddpm  --top_p 0.99 --warmup_steps 500 --logging_steps 50 --save_steps 500  --add_t5_tags --max_steps 100000  --save_total_limit 1  --load_best_model_at_end true --checkpoint_best_model  --greater_is_better true --eval_steps 500 --save_total_limit 1 



# TODO: how much we can decrease the steps for classification? lets go for 100 
# s-10 we did not do qqp
python run_glue.py --model_name_or_path roberta-large --dataset_name qqp --output_dir /net/nfs.cirrascale/s2-research/rabeehk/outputs/simplex_new/ours_glue_lr_3e-5_inference_steps_100/qqp --do_train --do_eval --do_predict --max_seq_length 128 --per_device_train_batch_size 64 --per_device_eval_batch_size 64 --evaluation_strategy steps --save_strategy steps   --report_to tensorboard --overwrite_output_dir --pad_to_max_length  --simplex_value 5 --num_diffusion_steps 5000 --num_inference_diffusion_steps 100 --conditional_generation seq2seq  --learning_rate 3e-5 --gradient_accumulation_steps 2 --lr_scheduler_type cosine --beta_schedule squaredcos_improved_ddpm  --top_p 0.99 --warmup_steps 500 --logging_steps 50 --save_steps 500  --add_t5_tags --max_steps 100000  --save_total_limit 1  --load_best_model_at_end true --checkpoint_best_model  --greater_is_better true --eval_steps 500


# Lets run above for 400 steps as well 
python run_glue.py --model_name_or_path roberta-large --dataset_name qqp --output_dir /net/nfs.cirrascale/s2-research/rabeehk/outputs/simplex_new/ours_glue_lr_3e-5_inference_steps_400/qqp --do_train --do_eval --do_predict --max_seq_length 128 --per_device_train_batch_size 64 --per_device_eval_batch_size 64 --evaluation_strategy steps --save_strategy steps   --report_to tensorboard --overwrite_output_dir --pad_to_max_length  --simplex_value 5 --num_diffusion_steps 5000 --num_inference_diffusion_steps 400 --conditional_generation seq2seq  --learning_rate 3e-5 --gradient_accumulation_steps 2 --lr_scheduler_type cosine --beta_schedule squaredcos_improved_ddpm  --top_p 0.99 --warmup_steps 500 --logging_steps 50 --save_steps 500  --add_t5_tags --max_steps 100000  --save_total_limit 1  --load_best_model_at_end true --checkpoint_best_model  --greater_is_better true --eval_steps 500





# Testing summarization baseline 
python -m torch.distributed.launch     --nproc_per_node 3 run_summarization.py     --model_name_or_path t5-small     --do_train     --do_eval     --dataset_name xsum     --dataset_config "3.0.0"  --source_prefix "summarize: "     --output_dir /tmp/tst-summarization     --per_device_train_batch_size=4     --per_device_eval_batch_size=4     --overwrite_output_dir     --predict_with_generate --eval_steps 10 --save_steps 10 

# summarization test for ours
python -m torch.distributed.launch --nproc_per_node 3 run_summarization.py --model_name_or_path roberta-large --do_train --do_eval --dataset_name xsum --dataset_config "3.0.0" --output_dir /net/nfs.cirrascale/s2-research/rabeehk/outputs/test --per_device_train_batch_size=12 --per_device_eval_batch_size=24 --overwrite_output_dir  --report_to tensorboard --eval_steps 1000  --max_steps 1000000 --max_eval_samples 96 --max_source_length 384 --max_target_length 128 --max_seq_length 512 --conditional_generation "seq2seq" --num_inference_diffusion_steps 2500 --evaluation_strategy steps --simplex_value 5 --num_diffusion_steps 5000 --lr_scheduler_type cosine --learning_rate 1e-4 --pad_to_max_length false --beta_schedule squaredcos_improved_ddpm --weight_decay 0.01 --top_p 0.99 --warmup_steps 2000 --logging_steps 50 --save_steps 1000 --eval_steps 10