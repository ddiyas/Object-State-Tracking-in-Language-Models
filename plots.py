import json
import matplotlib.pyplot as plt
import numpy as np

with open("results_gemma2b.json") as f:
    results_2b = json.load(f)
with open("results_gemma7b.json") as f:
    results_7b = json.load(f)


def accuracy_by(results, key, values):
    accs = []
    for v in values:
        subset = [r for r in results if r[key] == v]
        if subset:
            accs.append(sum(1 for r in subset if r["is_correct"]) / len(subset) * 100)
        else:
            accs.append(0)
    return accs


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Behavioral Results Across Story Type and Transfer Count", fontsize=14)

# plot 1: accuracy by story type
types = ["control", "distractor", "red_herring"]
labels = ["Control", "Distractor", "Red Herring"]
x = np.arange(len(types))
width = 0.35
ax1.bar(
    x - width / 2,
    accuracy_by(results_2b, "type", types),
    width,
    label="Gemma 2B",
    color="steelblue",
)
ax1.bar(
    x + width / 2,
    accuracy_by(results_7b, "type", types),
    width,
    label="Gemma 7B",
    color="darkorange",
)
ax1.set_xticks(x)
ax1.set_xticklabels(labels)
ax1.set_ylabel("Accuracy (%)")
ax1.set_title("Accuracy by Story Type")
ax1.axhline(y=16.7, color="gray", linestyle=":", label="Random chance")
ax1.legend()
ax1.grid(True, alpha=0.3, axis="y")

# plot 2: accuracy by num_transfers
transfers = [0, 1, 2, 3, 4]
ax2.plot(
    transfers,
    accuracy_by(results_2b, "num_transfers", transfers),
    marker="o",
    color="steelblue",
    label="Gemma 2B",
)
ax2.plot(
    transfers,
    accuracy_by(results_7b, "num_transfers", transfers),
    marker="o",
    color="darkorange",
    label="Gemma 7B",
)
ax2.axhline(y=16.7, color="gray", linestyle=":", label="Random chance")
ax2.set_xlabel("Number of Transfers")
ax2.set_ylabel("Accuracy (%)")
ax2.set_title("Accuracy by Number of Transfers")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("behavioral_results.png", dpi=150, bbox_inches="tight")
print("saved behavioral_results.png")
