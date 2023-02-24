# Train the models.

BASE_DIR="/home/lily/jt856/documents/simplex-diffusion/"
shared_params="--model_name_or_path roberta-large --per_device_train_batch_size 24 --per_device_eval_batch_size 6 --do_train --do_eval  --evaluation_strategy steps --eval_steps 1000 --report_to tensorboard --overwrite_output_dir --max_seq_length 256 --max_eval_samples 48 --simplex_value 5 --num_diffusion_steps 5000 --num_inference_diffusion_steps 1000 --lr_scheduler_type linear --learning_rate 1e-4 --pad_to_max_length --beta_schedule squaredcos_improved_ddpm --weight_decay 0.0 --tokenized_data_path processed_data/openwebtext_256_split/ --top_p 0.99 --max_steps 100000 --gradient_accumulation_steps 8 --warmup_steps 2000 --logging_steps 50 --save_steps 1000 --conditional_generation ul2"


params_for_length_50="--model_name_or_path roberta-large --per_device_train_batch_size 24 --per_device_eval_batch_size 24 --do_train --do_eval  --evaluation_strategy steps --eval_steps 1000 --report_to tensorboard --overwrite_output_dir --max_seq_length 50 --max_eval_samples 96 --simplex_value 5 --num_diffusion_steps 5000 --num_inference_diffusion_steps 1000 --lr_scheduler_type linear --learning_rate 3e-5 --pad_to_max_length --beta_schedule squaredcos_improved_ddpm --weight_decay 0.01 --tokenized_data_path processed_data/openwebtext_50_split/ --top_p 0.99 --max_steps 200000 --gradient_accumulation_steps 1 --warmup_steps 2000 --logging_steps 50 --save_steps 1000 --conditional_generation prefix_lm"

params_for_simple_data="--model_name_or_path roberta-large --per_device_train_batch_size 24 --per_device_eval_batch_size 6 --do_train --do_eval  --evaluation_strategy steps --eval_steps 1000 --report_to tensorboard --overwrite_output_dir --max_seq_length 50 --max_eval_samples 48 --simplex_value 5 --num_diffusion_steps 5000 --num_inference_diffusion_steps 1000 --lr_scheduler_type linear --learning_rate 1e-4 --pad_to_max_length --beta_schedule squaredcos_improved_ddpm --weight_decay 0.01  --top_p 0.99 --max_steps 60000 --gradient_accumulation_steps 1 --warmup_steps 2000 --logging_steps 50 --save_steps 1000     --train_file /home/lily/jt856/documents/diffusion/small_data/simple-train.txt --validation_file /home/lily/jt856/documents/diffusion/small_data/simple-test.txt"

DEBUG_PARAMS="--eval_steps 2 --num_inference_diffusion_steps 3 --per_device_train_batch_size 12 --max_eval_samples 6 --gradient_accumulation_steps 1"
PARAMS_FOR_LOCAL=" --save_total_limit 1 "

# Train the base model
# python -m torch.distributed.launch --nproc_per_node 8 run_mlm.py --output_dir $BASE_DIR"/outputs/paper_experiments/ul2" ${shared_params} ${PARAMS_FOR_LOCAL}


# Train the self-condition model
#python -m torch.distributed.launch --nproc_per_node 8 run_mlm.py --output_dir $BASE_DIR"/outputs/paper_experiments/ul2_self_conditioning_logits" ${shared_params}  --self_condition logits --per_device_train_batch_size 12  --gradient_accumulation_steps 16 ${PARAMS_FOR_LOCAL}

# self-condition with an MLP layer
# python -m torch.distributed.launch --nproc_per_node 8 run_mlm.py --output_dir $BASE_DIR"/outputs/paper_experiments/ul2_self_conditioning_logits_with_mlp" ${shared_params}  --self_condition logits --per_device_train_batch_size 12  --gradient_accumulation_steps 16 ${PARAMS_FOR_LOCAL} --self_condition_mlp_projection

# self-condition with addition
# python -m torch.distributed.launch --nproc_per_node 8 run_mlm.py --output_dir $BASE_DIR"/outputs/paper_experiments/ul2_self_conditioning_logits_with_addition" ${shared_params}  --self_condition "logits_addition" --per_device_train_batch_size 12  --gradient_accumulation_steps 16 ${PARAMS_FOR_LOCAL}

# self-condition with mean
# python -m torch.distributed.launch --nproc_per_node 8 run_mlm.py --output_dir $BASE_DIR"/outputs/paper_experiments/ul2_self_conditioning_logits_with_mean" ${shared_params}  --self_condition "logits_mean" --per_device_train_batch_size 12  --gradient_accumulation_steps 16 ${PARAMS_FOR_LOCAL}

# self-condition with max
# python -m torch.distributed.launch --nproc_per_node 8 run_mlm.py --output_dir $BASE_DIR"/outputs/paper_experiments/ul2_self_conditioning_logits_with_max" ${shared_params}  --self_condition "logits_max" --per_device_train_batch_size 12  --gradient_accumulation_steps 16 ${PARAMS_FOR_LOCAL}

# Classifier-free guidance
# python -m torch.distributed.launch --nproc_per_node 8 run_mlm.py --output_dir $BASE_DIR"/outputs/paper_experiments/ul2_self_conditioning_logits_with_guidance_scale_2" ${shared_params}  --self_condition logits --per_device_train_batch_size 12  --gradient_accumulation_steps 16 --guidance_scale 2 ${PARAMS_FOR_LOCAL}


# Classifier-free guidance with mlp self-conditioned
#python -m torch.distributed.launch --nproc_per_node 8 run_mlm.py --output_dir $BASE_DIR"/outputs/paper_experiments/ul2_self_conditioning_logits_with_mlp_with_guidance_scale_2" ${shared_params}  --self_condition logits --per_device_train_batch_size 12  --gradient_accumulation_steps 16 --guidance_scale 2 ${PARAMS_FOR_LOCAL} --self_condition_mlp_projection

# TODO: WHEN RUNNING on 8 GPUS adapt the number of processes and gradient_acc
# Classifier-free guidance with addition
# python -m torch.distributed.launch --nproc_per_node 4 run_mlm.py --output_dir $BASE_DIR"/outputs/paper_experiments/ul2_self_conditioning_logits_addition_with_guidance_scale_2" ${shared_params}  --self_condition logits_addition --per_device_train_batch_size 12  --gradient_accumulation_steps 32 --guidance_scale 2 ${PARAMS_FOR_LOCAL}


# DEBUG MODEL
#python  -m torch.distributed.launch --nproc_per_node 4 run_mlm.py --output_dir "/data/outputs/paper_experiments/debug" ${shared_params} ${DEBUG_PARAMS}   ${PARAMS_FOR_LOCAL} --eval_steps 400 --temperature 1.0  --tokenized_data_path  "/net/nfs.cirrascale/s2-research/rabeehk/simplex-diffusion/processed_data/openwebtext_256_split" --eval_steps 30 --compute_eval_loss_with_simplex True --self_condition "logits_mean" --ssdlm_optimizer --conditional_generation "ul2_variable" ${DEBUG_PARAMS} --evaluation_strategy steps --max_eval_samples 10 --save_steps 2 --save_steps 2 --save_checkpoints_on_s3

# Train on the simple data
python run_mlm.py ${params_for_simple_data} --output_dir $BASE_DIR"/outputs/paper_experiments/simple_data"    --line_by_line ${PARAMS_FOR_LOCAL}

# python run_mlm.py ${params_for_simple_data} --output_dir $BASE_DIR"/outputs/paper_experiments/simple_data_conditional"  --conditional_generation ul2  ${PARAMS_FOR_LOCAL}


# DEBUG MODEL trained on length=50 with prefix_lm.
#python -m torch.distributed.launch --nproc_per_node 4  run_mlm.py --output_dir $BASE_DIR"/outputs/paper_experiments/ul2_length_50_context_25" ${params_for_length_50} ${PARAMS_FOR_LOCAL} --eval_context_size 25

# Debug model for length=50
#python -m torch.distributed.launch --nproc_per_node 4  run_mlm.py --output_dir $BASE_DIR"/outputs/paper_experiments/debug_low_lr_ul2_length_50_context_25" ${params_for_length_50} ${PARAMS_FOR_LOCAL} --eval_context_size 25 --save_steps 500 --eval_steps 500  --compute_eval_loss_with_simplex True  --learning_rate 1e-5

# Train only with prefix_lm.
#python -m torch.distributed.launch --nproc_per_node 4 run_mlm.py --output_dir $BASE_DIR"/outputs/paper_experiments/prefix_lm_length_256" ${shared_params} ${PARAMS_FOR_LOCAL} --gradient_accumulation_steps 16 --conditional_generation "prefix_lm"

# Train only with prefix_lm and the ssdlm optimizer.
# python -m torch.distributed.launch --nproc_per_node 4 run_mlm.py --output_dir $BASE_DIR"/outputs/paper_experiments/prefix_lm_length_256_ssdlm_optimizer" ${shared_params} ${PARAMS_FOR_LOCAL} --gradient_accumulation_steps 16 --conditional_generation "prefix_lm" --ssdlm_optimizer

# Train the models on the new ul2 objective.
# python -m torch.distributed.launch --nproc_per_node 8 run_mlm.py --output_dir $BASE_DIR"/outputs/paper_experiments/ul2_variable_self_condition_mean_mix_before_weights" ${shared_params}  --per_device_train_batch_size 12  --gradient_accumulation_steps 16 ${PARAMS_FOR_LOCAL} --conditional_generation "ul2_variable"  --self_condition "logits_mean"  --self_condition_mix_before_weights true --compute_eval_loss_with_simplex true --resume_from_checkpoint "/net/nfs.cirrascale/s2-research/rabeehk/outputs/paper_experiments/ul2_variable_self_condition_mean_mix_before_weights/checkpoint-6000"


# Train the base model on the UL2 objective.
#python -m torch.distributed.launch --nproc_per_node 8 run_mlm.py --output_dir $BASE_DIR"/outputs/paper_experiments/ul2_variable_self_condition_mean_mix_before_weights_base_size" ${shared_params}  --per_device_train_batch_size 12  --gradient_accumulation_steps 16 ${PARAMS_FOR_LOCAL} --conditional_generation "ul2_variable"  --self_condition "logits_mean"  --self_condition_mix_before_weights true --compute_eval_loss_with_simplex true  --save_checkpoints_on_s3  --model_name_or_path "roberta-base"


# Base size model length=512.
#python -m torch.distributed.launch --nproc_per_node 8 run_mlm.py --output_dir $BASE_DIR"/outputs/paper_experiments/ul2_variable_self_condition_mean_mix_before_weights_base_size_length_512" ${shared_params}  --per_device_train_batch_size 12  --gradient_accumulation_steps 16 ${PARAMS_FOR_LOCAL} --conditional_generation "ul2_variable"  --self_condition "logits_mean"  --self_condition_mix_before_weights true --compute_eval_loss_with_simplex true  --save_checkpoints_on_s3  --model_name_or_path "roberta-base" --max_seq_length 512   --tokenized_data_path processed_data/openwebtext_512_split/


# python -m torch.distributed.launch --nproc_per_node 8 run_mlm.py --output_dir $BASE_DIR"/outputs/paper_experiments/ul2_variable_self_condition_mean_mix_before_weights_large_size_length_512" ${shared_params}  --per_device_train_batch_size 6  --gradient_accumulation_steps 32 ${PARAMS_FOR_LOCAL} --conditional_generation "ul2_variable"  --self_condition "logits_mean"  --self_condition_mix_before_weights true --compute_eval_loss_with_simplex true  --save_checkpoints_on_s3  --max_seq_length 512   --tokenized_data_path processed_data/openwebtext_512_split/
