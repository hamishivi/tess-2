python -m scripts.calculate_model_output_perplexity \
        --model_name_or_path outputs/llama/try3_ul2/checkpoint-32000 \
        --simplex_value 5 \
        --num_diffusion_steps 5000  \
        --num_inference_diffusion_steps 100 \
        --beta_schedule squaredcos_improved_ddpm \
        --top_p 0.99 \
        --self_condition "logits_mean" \
        --dataset_name c4 --streaming --dataset_config_name en
