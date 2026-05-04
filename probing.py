import os

os.environ["JOBLIB_TEMP_FOLDER"] = "/tmp"

import json
import time
import torch
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from collections import Counter

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    StratifiedShuffleSplit,
    cross_validate,
)
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_size",
    type=str,
    required=True,
    choices=[
        "gemma2b",
        "gemma9b",
        "llama3b",
        "phi4",
        "qwen3b",
        "llama8b",
        "mistral7b",
        "qwen7b",
        "qwen14b",
    ],
)
parser.add_argument(
    "--cv_folds", type=int, default=5, help="Number of CV folds when feasible."
)
parser.add_argument(
    "--cv_repeats", type=int, default=10, help="Repeats for RepeatedStratifiedKFold."
)
parser.add_argument(
    "--shuffle_splits",
    type=int,
    default=20,
    help="Splits for StratifiedShuffleSplit fallback.",
)
parser.add_argument(
    "--test_size",
    type=float,
    default=0.2,
    help="Test size for StratifiedShuffleSplit fallback.",
)
parser.add_argument(
    "--permute_n",
    type=int,
    default=0,
    help="Number of label permutations per layer (0 disables).",
)
parser.add_argument(
    "--seed", type=int, default=42, help="Random seed for CV + permutation."
)
args = parser.parse_args()

MODEL_SIZE = args.model_size

RESULTS_FILE = f"results_{MODEL_SIZE}.json"
ACTIVATIONS_FILE = f"activations_{MODEL_SIZE}.pt"
OUTPUT_PLOT = f"probe_accuracy_{MODEL_SIZE}.png"

VALID_LOCATIONS = ["shelf", "table", "bed", "floor", "desk", "counter"]

print(f"Loading results and activations for {MODEL_SIZE}")

with open(RESULTS_FILE) as f:
    results = json.load(f)

activations = torch.load(ACTIVATIONS_FILE, weights_only=True)

first_id = list(activations.keys())[0]
num_layers = activations[first_id].shape[0]
print(f"Number of layers: {num_layers}")

le = LabelEncoder()
le.fit(VALID_LOCATIONS)


def label_hist(labels):
    c = Counter(labels)
    return {k: c.get(k, 0) for k in VALID_LOCATIONS}


def min_class_count(y_int):
    bc = np.bincount(y_int, minlength=len(VALID_LOCATIONS))
    return int(bc.min()), bc


def make_probe():
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    C=0.1,
                    solver="lbfgs",
                    max_iter=5000,
                    class_weight="balanced",
                ),
            ),
        ]
    )


def choose_cv(y_int, k_folds, seed):
    mcc, _ = min_class_count(y_int)
    if mcc >= k_folds:
        return (
            RepeatedStratifiedKFold(
                n_splits=k_folds, n_repeats=args.cv_repeats, random_state=seed
            ),
            f"RepeatedStratifiedKFold({k_folds}x{args.cv_repeats})",
        )
    return (
        StratifiedShuffleSplit(
            n_splits=args.shuffle_splits, test_size=args.test_size, random_state=seed
        ),
        f"StratifiedShuffleSplit(n={args.shuffle_splits}, test={args.test_size})",
    )


def eval_probe_both(X, y_int, seed, k_folds):
    """Single cross_validate call returning accuracy + balanced_accuracy."""
    if len(X) == 0:
        return None, None, None, None, None
    cv, cv_name = choose_cv(y_int, k_folds=k_folds, seed=seed)
    probe = make_probe()
    res = cross_validate(
        probe,
        X,
        y_int,
        cv=cv,
        scoring=["accuracy", "balanced_accuracy"],
    )
    return (
        float(res["test_accuracy"].mean()),
        float(res["test_accuracy"].std()),
        float(res["test_balanced_accuracy"].mean()),
        float(res["test_balanced_accuracy"].std()),
        cv_name,
    )


def permutation_baseline(X, y_int, seed, k_folds, n_perm):
    if n_perm <= 0:
        return None, None
    rng = np.random.default_rng(seed)
    perm_scores = []
    for i in range(n_perm):
        y_perm = rng.permutation(y_int)
        _, _, m_bal, _, _ = eval_probe_both(
            X, y_perm, seed=seed + 1000 + i, k_folds=k_folds
        )
        if m_bal is not None:
            perm_scores.append(m_bal)
    if not perm_scores:
        return None, None
    return float(np.mean(perm_scores)), float(np.std(perm_scores))


# ------------------------------------------------------------
# Align results with activations
# ------------------------------------------------------------
usable = []
skipped_no_pred = skipped_no_act = skipped_bad_label = 0

for r in results:
    if r.get("predicted", None) is None:
        skipped_no_pred += 1
        continue
    sid = r["id"]
    if sid not in activations:
        skipped_no_act += 1
        continue
    if r.get("correct_answer", None) not in VALID_LOCATIONS:
        skipped_bad_label += 1
        continue
    usable.append(r)

print(f"Total results: {len(results)}")
print(f"Usable: {len(usable)}")
print(
    f"Skipped: no_pred={skipped_no_pred}, no_activation={skipped_no_act}, bad_label={skipped_bad_label}"
)

correct = [r for r in usable if r["is_correct"]]
incorrect = [r for r in usable if not r["is_correct"]]
print(f"Correct: {len(correct)} | Incorrect: {len(incorrect)}")

all_labels = [r["correct_answer"] for r in usable]
inc_labels = [r["correct_answer"] for r in incorrect]
print("\nLabel histogram (ALL usable):", label_hist(all_labels))
print("Label histogram (INCORRECT only):", label_hist(inc_labels))

model_accuracy = sum(1 for r in usable if r["is_correct"]) / max(1, len(usable))
chance_bal_acc = 1 / len(VALID_LOCATIONS)

print(f"\nRandom chance (balanced acc): {chance_bal_acc*100:.1f}%")
print(f"Model output accuracy (usable set): {model_accuracy*100:.1f}%")

# ------------------------------------------------------------
# Probe each layer
# ------------------------------------------------------------
print("\nTraining probes per layer...")

all_acc_mean, all_acc_std = [], []
all_bal_mean, all_bal_std = [], []
inc_acc_mean, inc_acc_std = [], []
inc_bal_mean, inc_bal_std = [], []
all_perm_bal_mean, all_perm_bal_std = [], []
inc_perm_bal_mean, inc_perm_bal_std = [], []

cv_used_all = cv_used_inc = None

for layer_idx in range(num_layers):
    t0 = time.time()

    # --- ALL usable ---
    X_all = np.stack(
        [activations[r["id"]][layer_idx].float().numpy() for r in usable], axis=0
    )
    y_all = le.transform([r["correct_answer"] for r in usable])

    m_acc, s_acc, m_bal, s_bal, cv_name_all = eval_probe_both(
        X_all, y_all, seed=args.seed + layer_idx, k_folds=args.cv_folds
    )
    all_acc_mean.append(m_acc)
    all_acc_std.append(s_acc)
    all_bal_mean.append(m_bal)
    all_bal_std.append(s_bal)
    cv_used_all = cv_used_all or cv_name_all

    p_m, p_s = permutation_baseline(
        X_all,
        y_all,
        seed=args.seed + 20_000 + layer_idx,
        k_folds=args.cv_folds,
        n_perm=args.permute_n,
    )
    all_perm_bal_mean.append(p_m)
    all_perm_bal_std.append(p_s)

    # --- INCORRECT only ---
    if len(incorrect) > 0:
        X_inc = np.stack(
            [activations[r["id"]][layer_idx].float().numpy() for r in incorrect], axis=0
        )
        y_inc = le.transform([r["correct_answer"] for r in incorrect])

        if len(np.unique(y_inc)) >= 2:
            m_acc_i, s_acc_i, m_bal_i, s_bal_i, cv_name_inc = eval_probe_both(
                X_inc, y_inc, seed=args.seed + 30_000 + layer_idx, k_folds=args.cv_folds
            )
            inc_acc_mean.append(m_acc_i)
            inc_acc_std.append(s_acc_i)
            inc_bal_mean.append(m_bal_i)
            inc_bal_std.append(s_bal_i)
            cv_used_inc = cv_used_inc or cv_name_inc

            p_mi, p_si = permutation_baseline(
                X_inc,
                y_inc,
                seed=args.seed + 50_000 + layer_idx,
                k_folds=args.cv_folds,
                n_perm=args.permute_n,
            )
            inc_perm_bal_mean.append(p_mi)
            inc_perm_bal_std.append(p_si)
        else:
            inc_acc_mean.append(None)
            inc_acc_std.append(None)
            inc_bal_mean.append(None)
            inc_bal_std.append(None)
            inc_perm_bal_mean.append(None)
            inc_perm_bal_std.append(None)
    else:
        inc_acc_mean.append(None)
        inc_acc_std.append(None)
        inc_bal_mean.append(None)
        inc_bal_std.append(None)
        inc_perm_bal_mean.append(None)
        inc_perm_bal_std.append(None)

    elapsed = time.time() - t0
    if layer_idx % 5 == 0:
        print(f"  layer {layer_idx}/{num_layers} done... ({elapsed:.1f}s)")

print("\nCV used (ALL):", cv_used_all)
print("CV used (INC):", cv_used_inc)


def best_layer(vals):
    arr = np.array([v if v is not None else -np.inf for v in vals], dtype=float)
    idx = int(np.argmax(arr))
    return idx, float(arr[idx])


best_all_layer, best_all_bal = best_layer(all_bal_mean)
print(
    f"\nBest probe (ALL) balanced acc: {best_all_bal*100:.1f}% at layer {best_all_layer}"
)

if any(v is not None for v in inc_bal_mean):
    best_inc_layer, best_inc_bal = best_layer(inc_bal_mean)
    print(
        f"Best probe (INC) balanced acc: {best_inc_bal*100:.1f}% at layer {best_inc_layer}"
    )

# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------
layers = np.arange(num_layers)


def to_np(lst):
    return np.array([np.nan if v is None else v for v in lst], dtype=float)


all_bal_m = to_np(all_bal_mean) * 100
all_bal_s = to_np(all_bal_std) * 100
inc_bal_m = to_np(inc_bal_mean) * 100
inc_bal_s = to_np(inc_bal_std) * 100
all_perm_m = to_np(all_perm_bal_mean) * 100
all_perm_s = to_np(all_perm_bal_std) * 100
inc_perm_m = to_np(inc_perm_bal_mean) * 100
inc_perm_s = to_np(inc_perm_bal_std) * 100

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(f"Probe Balanced Accuracy by Layer — {MODEL_SIZE}", fontsize=14)

ax1.plot(layers, all_bal_m, marker="o", color="steelblue", label="Probe (balanced acc)")
ax1.fill_between(
    layers,
    all_bal_m - all_bal_s,
    all_bal_m + all_bal_s,
    color="steelblue",
    alpha=0.2,
    linewidth=0,
)
if args.permute_n > 0 and not np.all(np.isnan(all_perm_m)):
    ax1.plot(
        layers,
        all_perm_m,
        color="black",
        linestyle="--",
        linewidth=1.0,
        label=f"Permuted labels (n={args.permute_n})",
    )
    ax1.fill_between(
        layers,
        all_perm_m - all_perm_s,
        all_perm_m + all_perm_s,
        color="black",
        alpha=0.1,
        linewidth=0,
    )
ax1.axhline(
    y=model_accuracy * 100,
    color="red",
    linestyle="--",
    label=f"Model output acc ({model_accuracy*100:.1f}%)",
)
ax1.axhline(
    y=chance_bal_acc * 100,
    color="gray",
    linestyle=":",
    label=f"Chance ({chance_bal_acc*100:.1f}%)",
)
ax1.set_xlabel("Layer")
ax1.set_ylabel("Balanced Accuracy (%)")
ax1.set_title("All Usable Stories")
ax1.legend()
ax1.grid(True, alpha=0.3)

valid_mask = ~np.isnan(inc_bal_m)
ax2.plot(
    layers[valid_mask],
    inc_bal_m[valid_mask],
    marker="o",
    color="darkorange",
    label="Probe (balanced acc, incorrect only)",
)
ax2.fill_between(
    layers[valid_mask],
    (inc_bal_m - inc_bal_s)[valid_mask],
    (inc_bal_m + inc_bal_s)[valid_mask],
    color="darkorange",
    alpha=0.2,
    linewidth=0,
)
if args.permute_n > 0 and not np.all(np.isnan(inc_perm_m)):
    ax2.plot(
        layers[valid_mask],
        inc_perm_m[valid_mask],
        color="black",
        linestyle="--",
        linewidth=1.0,
        label="Permuted labels",
    )
    ax2.fill_between(
        layers[valid_mask],
        (inc_perm_m - inc_perm_s)[valid_mask],
        (inc_perm_m + inc_perm_s)[valid_mask],
        color="black",
        alpha=0.1,
        linewidth=0,
    )
ax2.axhline(
    y=chance_bal_acc * 100,
    color="gray",
    linestyle=":",
    label=f"Chance ({chance_bal_acc*100:.1f}%)",
)
ax2.set_xlabel("Layer")
ax2.set_ylabel("Balanced Accuracy (%)")
ax2.set_title(
    "Incorrect Predictions Only\n(Decodable true label despite wrong output?)"
)
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_PLOT, dpi=150, bbox_inches="tight")
print(f"\nPlot saved to {OUTPUT_PLOT}")
