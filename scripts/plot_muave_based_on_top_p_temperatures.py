import json
import matplotlib.pyplot as plt
import os
import numpy as np

dir = "/net/nfs.cirrascale/s2-research/rabeehk/outputs/paper_experiments/tune_temperature"
top_ps = [0.1, 0.5, 0.7, 0.9, 0.99]
temperatures = [0.1, 0.5, 1.0, 2.0, 4.0, 10.0]

for k in ["pred_texts_from_logits_masked_muave"]: #, "pred_texts_from_simplex_masked_muave"]:
  for top_p in top_ps:
    muaves = []
    for temperature in temperatures:
        path = f"ul2_self_condition_addition_context_25_generations_{top_p}_temperature_{temperature}/eval_results.json"
        results = json.load(open(os.path.join(dir, path)))
        muave = results[k]
        muaves.append(np.round(muave*100, 2))
    
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
