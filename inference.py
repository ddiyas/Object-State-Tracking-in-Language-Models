import json
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_SIZE = "160m"  # using 160m , 1.4b, 6.9b

MODEL_NAME = f"EleutherAI/pythia-{MODEL_SIZE}"
RESULTS_FILE = f"results_{MODEL_SIZE}.json"
ACTIVATIONS_FILE = f"activations_{MODEL_SIZE}.pt"
DATASET_FILE = "dataset.json"

VALID_LOCATIONS = ["table", "shelf", "drawer", "backpack", "desk", "counter"]
TOP_K = 5  # check top 5 tokens for a valid location

print(f"Loading model: {MODEL_NAME}")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, output_hidden_states=True)
model.eval()
model.to(device)

print(f"Model loaded. Layers: {model.config.num_hidden_layers}")

with open(DATASET_FILE, "r") as f:
    dataset = json.load(f)

results = []
all_activations = {}  # story id : tensor of shape [num_layers, hidden_size]

for item in dataset:
    story_id = item["id"]
    obj = item["object"]
    correct_answer = item["answer"]

    # build prompt
    prompt = item["story"].replace("Where is the " + obj + "?", "").strip()
    prompt += f" The {obj} is on the"

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    # behavioral: grab top-k tokens and find first valid location
    logits = outputs.logits[0, -1, :]  # logits at last token position
    topk_token_ids = torch.topk(logits, TOP_K).indices.tolist()
    topk_tokens = [tokenizer.decode([tid]).strip().lower() for tid in topk_token_ids]

    predicted = None
    for token in topk_tokens:
        if token in VALID_LOCATIONS:
            predicted = token
            break

    is_correct = predicted == correct_answer

    print(
        f"[{story_id}] type={item['type']} transfers={item['num_transfers']} | "
        f"correct={correct_answer} predicted={predicted} ({'✓' if is_correct else '✗'})"
    )

    results.append(
        {
            "id": story_id,
            "type": item["type"],
            "num_transfers": item["num_transfers"],
            "num_location_distractors": item["num_location_distractors"],
            "num_random_distractors": item["num_random_distractors"],
            "object": obj,
            "correct_answer": correct_answer,
            "predicted": predicted,
            "top5_tokens": topk_tokens,
            "is_correct": is_correct,
            "prompt": prompt,
        }
    )

    # activations: grab residual stream at last token for every layer
    # hidden_states is a tuple of length num_layers + 1 (includes embedding layer)
    # each element is shape [batch, seq_len, hidden_size]
    # we want the last token position across all layers
    hidden_states = outputs.hidden_states  # tuple: (embedding, layer1, layer2, ...)

    # stack all layers except embedding: shape [num_layers, hidden_size]
    layer_activations = torch.stack(
        [h[0, -1, :] for h in hidden_states[1:]]  # skip embedding layer
    ).cpu()  # move to cpu to save memory

    all_activations[story_id] = layer_activations

# save results
with open(RESULTS_FILE, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nBehavioral results saved to {RESULTS_FILE}")

torch.save(all_activations, ACTIVATIONS_FILE)
print(f"Activations saved to {ACTIVATIONS_FILE}")

# quick summary
total = len(results)
correct = sum(1 for r in results if r["is_correct"])
no_prediction = sum(1 for r in results if r["predicted"] is None)
print(f"\n── Summary for {MODEL_NAME} ──")
print(f"Total stories: {total}")
print(f"Correct: {correct} ({100*correct/total:.1f}%)")
print(f"No valid location in top5: {no_prediction} ({100*no_prediction/total:.1f}%)")

# breakdown by type
for t in ["distractor", "red_herring", "reversal"]:
    subset = [r for r in results if r["type"] == t]
    if subset:
        acc = sum(1 for r in subset if r["is_correct"]) / len(subset)
        print(f"  {t}: {acc*100:.1f}% ({len(subset)} stories)")

# breakdown by num_transfers
print("\nAccuracy by num_transfers:")
for n in sorted(set(r["num_transfers"] for r in results)):
    subset = [r for r in results if r["num_transfers"] == n]
    acc = sum(1 for r in subset if r["is_correct"]) / len(subset)
    print(f"  {n} transfers: {acc*100:.1f}% ({len(subset)} stories)")
