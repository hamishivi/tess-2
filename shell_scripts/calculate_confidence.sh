
python -m scripts.confidence_over_steps \
        --model_name_or_path linear_cdcd/checkpoint-59000 \
        --simplex_value 5 \
        --num_diffusion_steps 5000  \
        --num_inference_diffusion_steps 1 10 100 \
        --beta_schedule squaredcos_improved_ddpm \
        --top_p 0.99 \
        --model_type "cdcd"
