python -m scripts.confidence_over_steps \
        --model_name_or_path tulu_mistral_200k \
        --simplex_value 5 \
        --num_diffusion_steps 5000  \
        --num_inference_diffusion_steps 100 \
        --beta_schedule squaredcos_improved_ddpm \
        --top_p 0.99 \
        --self_condition logits_mean \
        --self_condition_mix_before_weights \
        --is_causal false
