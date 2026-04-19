import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder

#  change this to "gemma2b" or "gemma7b" 
MODEL_SIZE = "gemma7b"

RESULTS_FILE = f"results_{MODEL_SIZE}.json"
ACTIVATIONS_FILE = f"activations_{MODEL_SIZE}.pt"
OUTPUT_PLOT = f"probe_accuracy_{MODEL_SIZE}.png"

VALID_LOCATIONS = ["shelf", "table", "basket", "floor", "desk", "counter"]

print(f"Loading results and activations for {MODEL_SIZE}...")

with open(RESULTS_FILE) as f:
    results = json.load(f)

activations = torch.load(ACTIVATIONS_FILE)  # dict: story_id -> [num_layers, hidden_size]

# build X and y
# X: activation vector at each layer for each story
# y: correct location label for each story

# figure out number of layers from first entry
first_id = list(activations.keys())[0]
num_layers = activations[first_id].shape[0]
print(f"Number of layers: {num_layers}")

# encode location labels as integers
le = LabelEncoder()
le.fit(VALID_LOCATIONS)

# align results with activations
# only keep stories where model had a valid prediction
valid_results = [r for r in results if r["predicted"] is not None]
print(f"Total stories with valid predictions: {len(valid_results)}")

# separate correct and incorrect predictions
correct_results = [r for r in valid_results if r["is_correct"]]
incorrect_results = [r for r in valid_results if not r["is_correct"]]
print(f"Correct: {len(correct_results)} | Incorrect: {len(incorrect_results)}")

# probe each layer
print("\nTraining probes per layer...")

# probe on ALL stories
all_probe_accuracies = []

# probe on only INCORRECT stories (the interesting case)
incorrect_probe_accuracies = []

for layer_idx in range(num_layers):
    #  all stories 
    X_all = []
    y_all = []
    for r in valid_results:
        story_id = r["id"]
        if story_id not in activations:
            continue
        vec = activations[story_id][layer_idx].float().numpy()
        X_all.append(vec)
        y_all.append(r["correct_answer"])

    X_all = np.array(X_all)
    y_all = le.transform(y_all)

    clf = LogisticRegression(max_iter=5000, C=1.0, solver='saga')
    scores = cross_val_score(clf, X_all, y_all, cv=5, scoring="accuracy")
    all_probe_accuracies.append(scores.mean())

    # incorrect stories only
    if len(incorrect_results) >= 5:  # need at least 5 for cross val
        X_inc = []
        y_inc = []
        for r in incorrect_results:
            story_id = r["id"]
            if story_id not in activations:
                continue
            vec = activations[story_id][layer_idx].float().numpy()
            X_inc.append(vec)
            y_inc.append(r["correct_answer"])

        X_inc = np.array(X_inc)
        y_inc = le.transform(y_inc)

        clf_inc = LogisticRegression(max_iter=5000, C=1.0, solver='saga')
        scores_inc = cross_val_score(clf_inc, X_inc, y_inc, cv=3, scoring="accuracy")
        incorrect_probe_accuracies.append(scores_inc.mean())
    else:
        incorrect_probe_accuracies.append(None)

    if layer_idx % 5 == 0:
        print(f"  layer {layer_idx}/{num_layers} done...")

print("\nDone! Probe accuracies per layer:")
for i, acc in enumerate(all_probe_accuracies):
    print(f"  layer {i}: {acc*100:.1f}%")

#  baseline: random chance 
chance = 1 / len(VALID_LOCATIONS)
print(f"\nRandom chance baseline: {chance*100:.1f}%")

#  overall model output accuracy 
model_accuracy = sum(1 for r in valid_results if r["is_correct"]) / len(valid_results)
print(f"Model output accuracy: {model_accuracy*100:.1f}%")
print(f"Best probe accuracy (all stories): {max(all_probe_accuracies)*100:.1f}% at layer {np.argmax(all_probe_accuracies)}")
if any(x is not None for x in incorrect_probe_accuracies):
    valid_inc = [(i, x) for i, x in enumerate(incorrect_probe_accuracies) if x is not None]
    best_inc_layer, best_inc_acc = max(valid_inc, key=lambda x: x[1])
    print(f"Best probe accuracy (incorrect stories only): {best_inc_acc*100:.1f}% at layer {best_inc_layer}")

#  plot 
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(f"Probe Accuracy by Layer — {MODEL_SIZE}", fontsize=14)

# plot 1: all stories
layers = list(range(num_layers))
ax1.plot(layers, [a * 100 for a in all_probe_accuracies], marker='o', color='steelblue', label='Probe accuracy')
ax1.axhline(y=model_accuracy * 100, color='red', linestyle='--', label=f'Model output accuracy ({model_accuracy*100:.1f}%)')
ax1.axhline(y=chance * 100, color='gray', linestyle=':', label=f'Random chance ({chance*100:.1f}%)')
ax1.set_xlabel("Layer")
ax1.set_ylabel("Accuracy (%)")
ax1.set_title("All Stories")
ax1.legend()
ax1.grid(True, alpha=0.3)

# plot 2: incorrect stories only
inc_accs = [x * 100 if x is not None else None for x in incorrect_probe_accuracies]
inc_layers = [i for i, x in enumerate(inc_accs) if x is not None]
inc_vals = [x for x in inc_accs if x is not None]

ax2.plot(inc_layers, inc_vals, marker='o', color='darkorange', label='Probe accuracy (incorrect stories)')
ax2.axhline(y=chance * 100, color='gray', linestyle=':', label=f'Random chance ({chance*100:.1f}%)')
ax2.set_xlabel("Layer")
ax2.set_ylabel("Accuracy (%)")
ax2.set_title("Incorrect Predictions Only\n(Does the model internally know the right answer?)")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_PLOT, dpi=150, bbox_inches='tight')
print(f"\nPlot saved to {OUTPUT_PLOT}")