import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from scipy import stats
import os

# ── config ────────────────────────────────────────────────────────────────────

MODELS = {
    "Gemma 2B": "gemma2b",
    "Gemma 9B": "gemma9b",
    "Llama 3B": "llama3b",
    "Llama 8B": "llama8b",
    "Qwen 3B": "qwen3b",
    "Qwen 7B": "qwen7b",
    "Qwen 14B": "qwen14b",
    "Phi-4": "phi4",
    "Mistral 7B": "mistral7b",
}

FAMILIES = {
    "Gemma": ["Gemma 2B", "Gemma 9B"],
    "Llama": ["Llama 3B", "Llama 8B"],
    "Qwen": ["Qwen 3B", "Qwen 7B", "Qwen 14B"],
    "Phi-4 / Mistral": ["Phi-4", "Mistral 7B"],
}

MODEL_COLORS = {
    "Gemma 2B": "#4C72B0",
    "Gemma 9B": "#1A3F6F",
    "Llama 3B": "#DD8452",
    "Llama 8B": "#8B4513",
    "Qwen 3B": "#55A868",
    "Qwen 7B": "#2E7D32",
    "Qwen 14B": "#1B4D1E",
    "Phi-4": "#C44E52",
    "Mistral 7B": "#8172B2",
}

CHANCE = 1 / 6  # 16.7%

RESULTS_DIR = "results"
PROBE_LAYERS_DIR = "probe_layers"
PROBE_SUMMARY_DIR = "probe_summary"
PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

# ── load data ─────────────────────────────────────────────────────────────────


def load_model_data(model_key):
    all_bal = np.load(f"{PROBE_LAYERS_DIR}/probe_layers_all_{model_key}.npy")
    inc_bal = np.load(f"{PROBE_LAYERS_DIR}/probe_layers_inc_{model_key}.npy")
    with open(f"{PROBE_SUMMARY_DIR}/probe_summary_{model_key}.json") as f:
        summary = json.load(f)
    with open(f"{RESULTS_DIR}/results_{model_key}.json") as f:
        results = json.load(f)
    return all_bal, inc_bal, summary, results


data = {}
for name, key in MODELS.items():
    try:
        all_bal, inc_bal, summary, results = load_model_data(key)
        data[name] = {
            "all_bal": all_bal,
            "inc_bal": inc_bal,
            "summary": summary,
            "results": results,
            "model_acc": summary["model_accuracy"],
            "num_layers": summary["num_layers"],
        }
        print(
            f"Loaded {name}: {summary['num_layers']} layers, model acc {summary['model_accuracy']*100:.1f}%"
        )
    except FileNotFoundError as e:
        print(f"WARNING: missing files for {name}: {e}")

# ── statistical tests ─────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("STATISTICAL SIGNIFICANCE TESTS")
print("=" * 70)

stat_results = []

for name, d in data.items():
    all_bal = d["all_bal"]
    inc_bal = d["inc_bal"]
    inc_bal_clean = inc_bal[~np.isnan(inc_bal)]
    model_acc = d["model_acc"]

    # claim 1: probe on incorrect stories > chance
    t1, p1 = stats.ttest_1samp(inc_bal_clean, CHANCE, alternative="greater")

    # claim 2: probe on all stories > model output accuracy
    t2, p2 = stats.ttest_1samp(all_bal, model_acc, alternative="greater")

    best_all = float(np.nanmax(all_bal)) * 100
    best_inc = float(np.nanmax(inc_bal_clean)) * 100
    mean_inc = float(np.nanmean(inc_bal_clean)) * 100

    stat_results.append(
        {
            "name": name,
            "model_acc": model_acc * 100,
            "best_all": best_all,
            "best_inc": best_inc,
            "mean_inc": mean_inc,
            "t1": t1,
            "p1": p1,
            "t2": t2,
            "p2": p2,
            "sig1": p1 < 0.05,
            "sig2": p2 < 0.05,
        }
    )

    print(f"\n{name}")
    print(f"  Model output accuracy:              {model_acc*100:.1f}%")
    print(f"  Best probe (all stories):           {best_all:.1f}%")
    print(f"  Best probe (incorrect only):        {best_inc:.1f}%")
    print(f"  Mean probe (incorrect only):        {mean_inc:.1f}%")
    print(
        f"  Claim 1 (inc probe > chance):       t={t1:.2f}, p={p1:.4f}  {'✓' if p1 < 0.05 else '✗'}"
    )
    print(
        f"  Claim 2 (probe > model output acc): t={t2:.2f}, p={p2:.4f}  {'✓' if p2 < 0.05 else '✗'}"
    )

print("\n" + "=" * 70)
print("SUMMARY TABLE")
print("=" * 70)
print(
    f"{'Model':<14} {'OutAcc':>7} {'BestAll':>8} {'BestInc':>8} {'MeanInc':>8} {'Cl1':>4} {'Cl2':>4}"
)
print("-" * 60)
for r in stat_results:
    print(
        f"{r['name']:<14} {r['model_acc']:>6.1f}% {r['best_all']:>7.1f}% "
        f"{r['best_inc']:>7.1f}% {r['mean_inc']:>7.1f}% "
        f"{'✓' if r['sig1'] else '✗':>4} {'✓' if r['sig2'] else '✗':>4}"
    )
print("\nCl1 = probe on incorrect stories significantly > chance (16.7%), p < 0.05")
print("Cl2 = probe on all stories significantly > model output accuracy, p < 0.05")

# ── plot 1: probe accuracy by layer, per architecture family ──────────────────

fig1, axes = plt.subplots(2, 2, figsize=(14, 10), sharey=True)
fig1.suptitle("Probe Balanced Accuracy by Layer — Per Architecture Family", fontsize=14)

for ax, (family, members) in zip(axes.flat, FAMILIES.items()):
    for name in members:
        if name not in data:
            continue
        d = data[name]
        all_bal = d["all_bal"] * 100
        layers = np.arange(len(all_bal))
        ax.plot(
            layers,
            all_bal,
            marker="o",
            markersize=3,
            linewidth=1.8,
            color=MODEL_COLORS[name],
            label=name,
        )
        ax.axhline(
            y=d["model_acc"] * 100,
            color=MODEL_COLORS[name],
            linestyle="--",
            linewidth=1,
            alpha=0.5,
        )

    ax.axhline(y=CHANCE * 100, color="gray", linestyle=":", linewidth=1, label="Chance")
    ax.set_title(family)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Balanced Accuracy (%)")
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/probe_by_family.png", dpi=150, bbox_inches="tight")
print(f"\nSaved {PLOTS_DIR}/probe_by_family.png")

# ── plot 2: incorrect stories probe by layer, per family ─────────────────────

fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10), sharey=True)
fig2.suptitle(
    "Probe Balanced Accuracy (Incorrect Stories Only) — Per Family", fontsize=14
)

for ax, (family, members) in zip(axes2.flat, FAMILIES.items()):
    for name in members:
        if name not in data:
            continue
        inc_bal = data[name]["inc_bal"] * 100
        layers = np.arange(len(inc_bal))
        valid = ~np.isnan(inc_bal)
        ax.plot(
            layers[valid],
            inc_bal[valid],
            marker="o",
            markersize=3,
            linewidth=1.8,
            color=MODEL_COLORS[name],
            label=name,
        )

    ax.axhline(y=CHANCE * 100, color="gray", linestyle=":", linewidth=1, label="Chance")
    ax.set_title(family)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Balanced Accuracy (%)")
    ax.set_ylim(0, 85)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/probe_inc_by_family.png", dpi=150, bbox_inches="tight")
print(f"Saved {PLOTS_DIR}/probe_inc_by_family.png")

# ── plot 3: summary bar — model acc vs best probe vs best inc probe ───────────

ORDER = [
    "Gemma 2B",
    "Gemma 9B",
    "Llama 3B",
    "Llama 8B",
    "Qwen 3B",
    "Qwen 7B",
    "Qwen 14B",
    "Phi-4",
    "Mistral 7B",
]
order_filtered = [m for m in ORDER if m in data]

x = np.arange(len(order_filtered))
width = 0.25

model_accs = [data[m]["model_acc"] * 100 for m in order_filtered]
best_alls = [float(np.nanmax(data[m]["all_bal"])) * 100 for m in order_filtered]
best_incs = [
    float(np.nanmax(data[m]["inc_bal"][~np.isnan(data[m]["inc_bal"])])) * 100
    for m in order_filtered
]

fig3, ax3 = plt.subplots(figsize=(14, 6))
ax3.bar(
    x - width,
    model_accs,
    width,
    label="Model output accuracy",
    color="steelblue",
    alpha=0.85,
)
ax3.bar(
    x,
    best_alls,
    width,
    label="Best probe (all stories)",
    color="darkorange",
    alpha=0.85,
)
ax3.bar(
    x + width,
    best_incs,
    width,
    label="Best probe (incorrect only)",
    color="seagreen",
    alpha=0.85,
)

ax3.axhline(
    y=CHANCE * 100, color="gray", linestyle=":", linewidth=1, label="Random chance"
)

# family separators
for sep in [1.5, 3.5, 6.5]:
    ax3.axvline(x=sep, color="lightgray", linestyle="--", linewidth=0.8)

ax3.set_xticks(x)
ax3.set_xticklabels(order_filtered, rotation=25, ha="right")
ax3.set_ylabel("Balanced Accuracy (%)")
ax3.set_title("Model Output Accuracy vs Probe Accuracy — All Models")
ax3.set_ylim(0, 100)
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/probe_summary_bars.png", dpi=150, bbox_inches="tight")
print(f"Saved {PLOTS_DIR}/probe_summary_bars.png")

print("\nDone!")
