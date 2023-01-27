import json
import numpy as np

def read_results(path, is_baseline, key=None):
   prefix="test_pred_texts_from_logits_masked_" if not is_baseline else "predict_"
   #metrics = ["sari", "bleu", "bert_score",  "rouge1", "rouge2", "rougelsum", "dist-1", "dist-4"] # "bert_score_them", 
   # metrics = ["bleu", "bert_score_them",  "rougeLsum", "dist-1", "dist-4"]
   metrics = ["rouge1", "rouge2", "rougeLsum"]
   results = json.load(open(path, "r"))
   values = []
   results_metrics = []
   for metric in metrics:
     if f"{prefix}{metric}" in results:
       values.append(np.round(results[f"{prefix}{metric}"], 2))
       results_metrics.append(metric)

   values = [str(v) for v in values]
   #print("&".join(results_metrics))
   if key is not None:
      values = [key]+values
      print("&".join(values), r"\\")
   else:
      print("&".join(values), r"\\")


is_baseline=False
#path="/net/nfs.cirrascale/s2-research/rabeehk/outputs/paper_experiments/simplification_results/ours_lr_2e-5_no_wd/test_top_p_None_temperature_1.0_results.json"
# path = "/net/nfs.cirrascale/s2-research/rabeehk/outputs/paper_experiments/wiki_alignment_tune_lr/ours_lr_3e-5_no_wd/test_top_p_None_temperature_1.0_results.json"
# path="/net/nfs.cirrascale/s2-research/rabeehk/outputs/paper_experiments/wiki_alignment_tune_lr/roberta_base_lr_3e-5_no_wd/predict_top_p_None_temperature_1.0_results.json"
#path="/net/nfs.cirrascale/s2-research/rabeehk/outputs/paper_experiments/qqp_tune_steps/ours_lr_3e-5_steps_90000/checkpoint-90000/test_top_p_None_temperature_1.0_results.json"

#path="/net/nfs.cirrascale/s2-research/rabeehk/outputs/paper_experiments/wiki_alignment_self_condition_ablations/lr_3e-5_no_wd_steps_60000_no_self_condition/test_results.json"
#path="/net/nfs.cirrascale/s2-research/rabeehk/outputs/paper_experiments/wiki_alignment_self_condition_ablations/lr_3e-5_no_wd_steps_60000_self_condition_logits/test_results.json"
#path="/net/nfs.cirrascale/s2-research/rabeehk/outputs/paper_experiments/wiki_alignment_self_condition_ablations/lr_3e-5_no_wd_steps_60000_self_condition_logits_mean_ours_mix_before_weights/test_results.json"
#path="/net/nfs.cirrascale/s2-research/rabeehk/outputs/paper_experiments/qqp_tune_steps/bart_base_lr_3e-5_no_wd_max_steps_90000/predict_top_p_None_temperature_1.0_results.json"

# ablation results.
#path="/net/nfs.cirrascale/s2-research/rabeehk/outputs/paper_experiments/wiki_alignment_self_condition_ablations/lr_3e-5_no_wd_steps_60000_no_self_condition/checkpoint-60000/test_top_p_None_temperature_1.0_results.json"
#path="/net/nfs.cirrascale/s2-research/rabeehk/outputs/paper_experiments/wiki_alignment_self_condition_ablations/lr_3e-5_no_wd_steps_60000_self_condition_logits/checkpoint-60000/test_top_p_None_temperature_1.0_results.json"
#path="/net/nfs.cirrascale/s2-research/rabeehk/outputs/paper_experiments/wiki_alignment_self_condition_ablations/lr_3e-5_no_wd_steps_60000_self_condition_logits_mean_ours_mix_before_weights/checkpoint-60000/test_top_p_None_temperature_1.0_results.json"
#path="/net/nfs.cirrascale/s2-research/rabeehk/outputs/paper_experiments/qg_tune_steps/ours_lr_3e-5_steps_120000/test_top_p_None_temperature_1.0_seed_42_results.json"
#path="/net/nfs.cirrascale/s2-research/rabeehk/outputs/paper_experiments/simplification_results/baseline_lr_2e-5_no_wd/predict_top_p_None_temperature_1.0_results.json"
#path="/net/nfs.cirrascale/s2-research/rabeehk/outputs/paper_experiments/cnn_dailymail_results/ours_lr_3e-5_max_steps_120000_model_roberta-base/test_top_p_None_temperature_1.0_seed_42_results.json"
# path="/net/nfs.cirrascale/s2-research/rabeehk/outputs/paper_experiments/cnn_dailymail_results/baseline_bart_base_lr_3e-5_max_steps_120000/predict_top_p_None_temperature_1.0_seed_42_results.json"

# guidance scores.
'''
paths = {}
for t in ["1.5", "10.0", "16.0", "20.0", "40.0", "100.0",  "500.0"]: #["2.0", "4.0", "8.0", "10.0", "16.0"]:
   # paths[t]=f"/net/nfs.cirrascale/s2-research/rabeehk/outputs/paper_experiments/wiki_alignment_with_guidance/ours_lr_3e-5_no_wd_guidance_2.0_inputs_empty_token_repeat/checkpoint-80000/examples_500/test_top_p_None_temperature_1.0_seed_42_guidance_scale_{t}_results.json"
   # paths[t]=f"/net/nfs.cirrascale/s2-research/rabeehk/outputs/paper_experiments/wiki_alignment_with_guidance/ours_lr_3e-5_no_wd_guidance_2.0_inputs_empty_token_repeat/checkpoint-80000/examples_500/softmax/test_top_p_None_temperature_1.0_seed_42_guidance_scale_{t}_results.json"
   
   # paths[t] = f"/net/nfs.cirrascale/s2-research/rabeehk/outputs/paper_experiments/wiki_alignment_with_guidance/ours_lr_3e-5_no_wd_guidance_2.0_mask_token/checkpoint-80000/examples_500/test_top_p_None_temperature_1.0_seed_42_guidance_scale_{t}_results.json"
   # paths[t] = f"/net/nfs.cirrascale/s2-research/rabeehk/outputs/paper_experiments/wiki_alignment_with_guidance/ours_lr_3e-5_no_wd_guidance_2.0_mask_token/checkpoint-80000/examples_500/softmax/test_top_p_None_temperature_1.0_seed_42_guidance_scale_{t}_results.json"

   #paths[t] = f"/net/nfs.cirrascale/s2-research/rabeehk/outputs/paper_experiments/wiki_alignment_with_guidance/ours_lr_3e-5_no_wd_guidance_2.0_mask_token_inputs/checkpoint-80000/examples_500/test_top_p_None_temperature_1.0_seed_42_guidance_scale_{t}_results.json"
   # paths[t] = f"/net/nfs.cirrascale/s2-research/rabeehk/outputs/paper_experiments/wiki_alignment_with_guidance/ours_lr_3e-5_no_wd_guidance_2.0_mask_token_inputs/checkpoint-80000/examples_500/softmax/test_top_p_None_temperature_1.0_seed_42_guidance_scale_{t}_results.json"
   
   # paths[t] = f"/net/nfs.cirrascale/s2-research/rabeehk/outputs/paper_experiments/wiki_alignment_with_guidance/ours_lr_3e-5_no_wd_guidance_2.0/checkpoint-80000/examples_500/test_top_p_None_temperature_1.0_seed_42_guidance_scale_{t}_results.json"
   # paths[t] = f"/net/nfs.cirrascale/s2-research/rabeehk/outputs/paper_experiments/wiki_alignment_with_guidance/ours_lr_3e-5_no_wd_guidance_2.0/checkpoint-80000/examples_500/softmax/test_top_p_None_temperature_1.0_seed_42_guidance_scale_{t}_results.json"
   # paths[t]=f"/net/nfs.cirrascale/s2-research/rabeehk/outputs/paper_experiments/wiki_alignment_tune_lr/ours_lr_3e-5_no_wd/checkpoint-80000/examples_500/test_top_p_None_temperature_1.0_seed_42_guidance_scale_1.0_results.json"
   paths[t] =  f"/net/nfs.cirrascale/s2-research/rabeehk/outputs/paper_experiments/wiki_alignment_with_guidance/ours_lr_3e-5_no_wd_guidance_2.0/checkpoint-80000/examples_500/softmax/test_top_p_None_temperature_1.0_seed_42_guidance_scale_{t}_results.json"

for k, path in paths.items():
   #print(k) 
   read_results(path, is_baseline, key=k)
'''
'''
# qqp ablation.
paths = {}
top_p="0.99"
paths.update({"no-self": f"/net/nfs.cirrascale/s2-research/rabeehk/outputs/paper_experiments/qqp_self_condition_ablations/ours_lr_3e-5_steps_60000_no_self/checkpoint-60000/test_top_p_{top_p}_temperature_1.0_seed_42_guidance_scale_1.0_results.json"})
paths.update({"logits": f"/net/nfs.cirrascale/s2-research/rabeehk/outputs/paper_experiments/qqp_self_condition_ablations/ours_lr_3e-5_steps_60000_self_logits/checkpoint-60000/test_top_p_{top_p}_temperature_1.0_seed_42_guidance_scale_1.0_results.json"})
paths.update({"ours": f"/net/nfs.cirrascale/s2-research/rabeehk/outputs/paper_experiments/qqp_self_condition_ablations/ours_lr_3e-5_steps_60000_self_condition_logits_mean_mix_before_weights/checkpoint-60000/test_top_p_{top_p}_temperature_1.0_seed_42_guidance_scale_1.0_results.json"})
        
for k, path in paths.items():
   print(k) 
   read_results(path, is_baseline)
'''

# qqp less iterations.
#path="/net/nfs.cirrascale/s2-research/rabeehk/outputs/paper_experiments/qqp_tune_steps/ours_lr_3e-5_steps_90000/checkpoint-90000/ablation_inference_steps/steps_100/test_top_p_None_temperature_1.0_seed_42_guidance_scale_1.0_results.json"
#read_results(path, False)


# newsela less iterations.
#path="/net/nfs.cirrascale/s2-research/rabeehk/outputs/paper_experiments/wiki_alignment_tune_lr/ours_lr_3e-5_no_wd/checkpoint-80000/inference_steps_ablation_all_data/step_10"
#read_results(path, False)


# qg dataset.
# path="/net/nfs.cirrascale/s2-research/rabeehk/outputs/paper_experiments/qg_tune_steps/ours_lr_3e-5_steps_120000/checkpoint-120000/inference_steps_ablations/steps_100/test_top_p_None_temperature_1.0_seed_42_guidance_scale_1.0_results.json"
# read_results(path, False)

# text summarization.
path="/net/nfs.cirrascale/s2-research/rabeehk/outputs/paper_experiments/cnn_dailymail_results/ours_lr_3e-5_max_steps_120000_model_roberta-base/checkpoint-120000/inference_steps_ablation_on_all_data/step_100/test_top_p_None_temperature_1.0_seed_42_guidance_scale_1.0_results.json"
read_results(path, False)
