import json
import matplotlib.pyplot as plt
import numpy as np

# Map: label -> results_*.json filename
MODELS = {
    "Gemma 2B": "results_gemma2b.json",
    "Gemma 9B": "results_gemma9b.json",
    "Llama 3B": "results_llama3b.json",
    "Llama 8B": "results_llama8b.json",
    "Qwen 3B": "results_qwen3b.json",
    "Qwen 7B": "results_qwen7b.json",
    "Qwen 14B": "results_qwen14b.json",
    "Phi-4": "results_phi4.json",
    "Mistral 7B": "results_mistral7b.json",
}

def load_results(path):
    with open(path) as f:
        return json.load(f)

results_by_model = {name: load_results(path) for name, path in MODELS.items()}

def accuracy_by(results, key, values):
    """Returns accuracy% for each v in values. If no examples for v, returns NaN."""
    accs = []
    for v in values:
        subset = [r for r in results if r.get(key) == v]
        if subset:
            accs.append(sum(1 for r in subset if r.get("is_correct")) / len(subset) * 100.0)
        else:
            accs.append(np.nan)
    return np.array(accs, dtype=float)

# --- Figure setup ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
fig.suptitle("Behavioral Results Across Story Type and Transfer Count", fontsize=14)

# Use a consistent color cycle across plots
model_names = list(results_by_model.keys())
colors = plt.cm.tab10(np.linspace(0, 1, len(model_names)))  # tab10 is fine for up to 10 lines

# Plot 1: accuracy by story type (grouped bars)
types = ["control", "distractor", "red_herring"]
type_labels = ["Control", "Distractor", "Red Herring"]

x = np.arange(len(types))
n_models = len(model_names)
group_width = 0.86
bar_width = group_width / n_models

for i, (model, color) in enumerate(zip(model_names, colors)):
    vals = accuracy_by(results_by_model[model], "type", types)
    # center the bars around each x tick
    offsets = x - group_width / 2 + (i + 0.5) * bar_width
    ax1.bar(offsets, vals, bar_width, label=model, color=color, alpha=0.9)

ax1.set_xticks(x)
ax1.set_xticklabels(type_labels)
ax1.set_ylabel("Accuracy (%)")
ax1.set_title("Accuracy by Story Type")
ax1.axhline(y=100.0 / 6.0, color="gray", linestyle=":", label="Random chance (1/6)")
ax1.grid(True, alpha=0.3, axis="y")
ax1.legend(fontsize=9, ncol=2)

# Plot 2: accuracy by number of transfers (lines)
transfers = [0, 1, 2, 3, 4]

for model, color in zip(model_names, colors):
    vals = accuracy_by(results_by_model[model], "num_transfers", transfers)
    ax2.plot(transfers, vals, marker="o", linewidth=2, color=color, label=model)

ax2.axhline(y=100.0 / 6.0, color="gray", linestyle=":", label="Random chance (1/6)")
ax2.set_xlabel("Number of Transfers")
ax2.set_ylabel("Accuracy (%)")
ax2.set_title("Accuracy by Number of Transfers")
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=9, ncol=2)

plt.tight_layout()
plt.savefig("behavioral_results.png", dpi=150, bbox_inches="tight")
print("saved behavioral_results.png")