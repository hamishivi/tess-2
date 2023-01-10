import json
import matplotlib.pyplot as plt

top_ps = [0.0, 0.1, 0.2, 0.5, 0.7, 0.9, 0.95, 0.99]
muaves = []
for top_p in top_ps:
    path = f"/net/nfs.cirrascale/s2-research/rabeehk/outputs/paper_experiments/tune_top_p/ul2_length_50_context_25_generations_{top_p}/eval_results.json"
    metrics = json.load(open(path))
    muave = metrics["pred_texts_from_simplex_masked_muave"]
    muaves.append(muave)

    print(muaves)
    print(top_ps)

# Plot
plt.plot(top_ps, muaves)
plt.xlabel("top-p")
plt.ylabel("muave")
plt.savefig("image.png")
