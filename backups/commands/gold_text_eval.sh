# computing gold text metrics.

truncation_length=206
output_dir="/net/nfs.cirrascale/s2-research/rabeehk/outputs/paper_experiments/gold_truncation_"${truncation}"_context_size_25"
python gold_text_eval.py  --model_name_or_path "gpt2-large" --output_dir "gold_evals" --tokenized_data_path processed_data/openwebtext_256_split_gpt_eval/ --max_eval_samples 1000 --conditional_generation "prefix_lm" --truncation_length ${truncation_length}  --eval_context_size 25  --eval_for_all_metrics --max_seq_length 256 --output_dir ${output_dir}
