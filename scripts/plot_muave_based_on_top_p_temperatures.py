import json
import matplotlib.pyplot as plt
import os
import numpy as np

top_ps = [0.9, 0.95, 0.99] #[0.1, 0.5, 0.7, 0.9, 0.99]
temperatures = [1.0, 2.0, 4.0, 10.0] # 0.1, 0.5

for k in ["pred_texts_from_logits_masked_muave"]: #, "pred_texts_from_simplex_masked_muave"]:
  for top_p in top_ps:
    muaves = []
    actual_ts = []
    for temperature in temperatures:
        path=f"/net/nfs.cirrascale/s2-research/rabeehk/outputs/paper_experiments/tune_temperatures/ul2_self_condition_addition_context_25_top_p_{top_p}_temperature_{temperature}_truncation_56/checkpoint-15000/eval_results.json"
        # path=f"/net/nfs.cirrascale/s2-research/rabeehk/outputs/paper_experiments/tune_temperatures/ul2_self_condition_context_25_top_p_{top_p}_temperature_{temperature}_truncation_56/checkpoint-15000/eval_results.json"
        # path=f"/net/nfs.cirrascale/s2-research/rabeehk/outputs/paper_experiments/tune_temperatures/ul2_context_25_top_p_{top_p}_temperature_{temperature}_truncation_56/checkpoint-19000/eval_results.json"
        # path = f"/net/nfs.cirrascale/s2-research/rabeehk/outputs/paper_experiments/tune_temperature/ul2_self_condition_addition_context_25_generations_{top_p}_temperature_{temperature}/eval_results.json"
        results = json.load(open(path))
        if k in results:
          muave = results[k]
          muaves.append(np.round(muave*100, 2))
          actual_ts.append(temperature)
          print(muaves) 
          print(actual_ts)
    print("P ", top_p)
    print("Temperatures ", temperatures) 
    print("MUAVEs       ", muaves)
    # Plot it.
    #plt.plot(temperature, muaves)
    #plt.title(f"top_p_{top_p}")
    #plt.xlabel(f"temperature")
    #plt.ylabel("muave")
    #plt.savefig(f"temperature_top_p_{top_p}.png")

  print("="*100)
