

python -m scripts.interactive_demo \
        --model_name_or_path tmp/checkpoint-48000 \
        --simplex_value 5 \
        --num_diffusion_steps 5000  \
        --num_inference_diffusion_steps 1 10 100 \
        --beta_schedule squaredcos_improved_ddpm \
        --top_p 0.99 \
        --self_condition "logits_mean" \
        --self_condition_mix_before_weights \
        --token_warp