import json
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

MODELS = {
    "Gemma 2B": "results\\results_gemma2b.json",
    "Gemma 9B": "results\\results_gemma9b.json",
    "Llama 3B": "results\\results_llama3b.json",
    "Llama 8B": "results\\results_llama8b.json",
    "Qwen 3B": "results\\results_qwen3b.json",
    "Qwen 7B": "results\\results_qwen7b.json",
    "Qwen 14B": "results\\results_qwen14b.json",
    "Phi-4-Mini": "results\\results_phi4.json",
    "Mistral 7B": "results\\results_mistral7b.json",
}

FAMILIES = {
    "Gemma": ["Gemma 2B", "Gemma 9B"],
    "Llama": ["Llama 3B", "Llama 8B"],
    "Qwen": ["Qwen 3B", "Qwen 7B", "Qwen 14B"],
    "Phi-4-Mini / Mistral": ["Phi-4-Mini", "Mistral 7B"],
}

# color per model, consistent across both figures
MODEL_COLORS = {
    "Gemma 2B": "#4C72B0",
    "Gemma 9B": "#1A3F6F",
    "Llama 3B": "#DD8452",
    "Llama 8B": "#8B4513",
    "Qwen 3B": "#55A868",
    "Qwen 7B": "#2E7D32",
    "Qwen 14B": "#1B4D1E",
    "Phi-4-Mini": "#C44E52",
    "Mistral 7B": "#8172B2",
}

TRANSFERS = [0, 1, 2, 3, 4]
CHANCE = 100.0 / 6.0


def load(path):
    with open(path) as f:
        return json.load(f)


results_by_model = {name: load(path) for name, path in MODELS.items()}


def acc_by(results, key, values):
    out = []
    for v in values:
        subset = [r for r in results if r.get(key) == v]
        out.append(
            sum(1 for r in subset if r.get("is_correct")) / len(subset) * 100.0
            if subset
            else np.nan
        )
    return np.array(out, dtype=float)


# ── Figure 1: transfer curves by architecture family ──────────────────────────
fig1, axes = plt.subplots(2, 2, figsize=(14, 10), sharey=True)
fig1.suptitle("Accuracy by Number of Transfers — Per Architecture Family", fontsize=14)

for ax, (family, members) in zip(axes.flat, FAMILIES.items()):
    for model in members:
        vals = acc_by(results_by_model[model], "num_transfers", TRANSFERS)
        ax.plot(
            TRANSFERS,
            vals,
            marker="o",
            linewidth=2,
            color=MODEL_COLORS[model],
            label=model,
        )
    ax.axhline(
        y=CHANCE, color="gray", linestyle=":", linewidth=1, label="Random chance"
    )
    ax.set_title(family)
    ax.set_xlabel("Number of Transfers")
    ax.set_ylabel("Accuracy (%)")
    ax.set_xticks(TRANSFERS)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig("behavioral_by_family.png", dpi=150, bbox_inches="tight")
print("saved behavioral_by_family.png")

# ── Figure 2: summary bar chart ordered by family then size ───────────────────
ORDER = [
    "Gemma 2B",
    "Gemma 9B",
    "Llama 3B",
    "Llama 8B",
    "Qwen 3B",
    "Qwen 7B",
    "Qwen 14B",
    "Phi-4-Mini",
    "Mistral 7B",
]

TYPES = ["control", "distractor", "red_herring"]
T_LABEL = ["Control", "Distractor", "Red Herring"]

fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
fig2.suptitle("Accuracy by Story Type — All Models", fontsize=14)

x = np.arange(len(ORDER))
bar_width = 0.6

for ax, t, label in zip(axes2, TYPES, T_LABEL):
    vals = [acc_by(results_by_model[m], "type", [t])[0] for m in ORDER]
    colors = [MODEL_COLORS[m] for m in ORDER]
    bars = ax.bar(x, vals, bar_width, color=colors, alpha=0.9, edgecolor="white")
    ax.axhline(
        y=CHANCE, color="gray", linestyle=":", linewidth=1, label="Random chance"
    )
    ax.set_title(label)
    ax.set_xticks(x)
    ax.set_xticklabels(ORDER, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3, axis="y")

    # add value labels on bars
    for bar, val in zip(bars, vals):
        if not np.isnan(val):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                f"{val:.0f}%",
                ha="center",
                va="bottom",
                fontsize=7.5,
            )

    # family separators
    for sep in [1.5, 3.5, 6.5]:
        ax.axvline(x=sep, color="lightgray", linestyle="--", linewidth=0.8)

# shared legend for family groupings
family_labels = ["Gemma", "Llama", "Qwen", "Phi-4-Mini / Mistral"]
from matplotlib.patches import Patch

legend_handles = [
    Patch(color="#4C72B0", label="Gemma"),
    Patch(color="#DD8452", label="Llama"),
    Patch(color="#55A868", label="Qwen"),
    Patch(color="#C44E52", label="Phi-4-Mini / Mistral"),
]
fig2.legend(
    handles=legend_handles,
    loc="lower center",
    ncol=4,
    fontsize=9,
    bbox_to_anchor=(0.5, -0.02),
)

plt.tight_layout()
plt.savefig("behavioral_by_type.png", dpi=150, bbox_inches="tight")
print("saved behavioral_by_type.png")
