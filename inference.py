import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"

import sys
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

try:
    import torch._dynamo
    torch._dynamo.disable()
except Exception:
    pass

MODEL_CONFIGS = {
    "gemma2b": "google/gemma-2-2b-it",
    "gemma9b": "google/gemma-2-9b-it",
    "gemma27b": "google/gemma-2-27b-it",
    "llama3b": "meta-llama/Llama-3.2-3B-Instruct",
    "llama8b": "meta-llama/Llama-3.1-8B-Instruct",
    "llama70b": "meta-llama/Llama-3.1-70B-Instruct",
    "qwen3b": "Qwen/Qwen2.5-3B-Instruct",
    "qwen7b": "Qwen/Qwen2.5-7B-Instruct",
    "qwen14b": "Qwen/Qwen2.5-14B-Instruct",
    "qwen32b": "Qwen/Qwen2.5-32B-Instruct",
    "phi4": "microsoft/Phi-4-mini-instruct",
    "mistral7b": "mistralai/Mistral-7B-Instruct-v0.3",
}

VALID_LOCATIONS = ["shelf", "table", "bed", "floor", "desk", "counter"]

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 inference.py <model_size>")
        print(f"Options: {', '.join(MODEL_CONFIGS.keys())}")
        sys.exit(1)

    model_size = sys.argv[1]
    if model_size not in MODEL_CONFIGS:
        print(f"Unknown model: {model_size}")
        print(f"Available: {list(MODEL_CONFIGS.keys())}")
        sys.exit(1)

    model_name = MODEL_CONFIGS[model_size]
    results_file = f"results_{model_size}.json"
    activations_file = f"activations_{model_size}.pt"
    dataset_file = "dataset.json"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model: {model_name}")
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        output_hidden_states=True,
        attn_implementation="eager",
    )
    model.eval()
    model.config.pad_token_id = tokenizer.pad_token_id

    print("Model loaded.")

    with open(dataset_file, "r") as f:
        dataset = json.load(f)

    results = []
    all_activations = {}  # story_id -> [num_layers, hidden_size] CPU tensor

    for item in dataset:
        story_id = item["id"]
        correct_answer = item["answer"]

        messages = [{
            "role": "user",
            "content": item["story"] + "\nAnswer with only the location word, nothing else."
        }]

        prompt_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            return_tensors="pt",
            add_generation_prompt=True,
        ).to(device)

        # Some tokenizers return a BatchEncoding; normalize to tensor
        if hasattr(prompt_ids, "input_ids"):
            prompt_ids = prompt_ids.input_ids

        with torch.inference_mode():
            outputs = model(input_ids=prompt_ids, output_hidden_states=True)

            output_ids = model.generate(
                input_ids=prompt_ids,
                max_new_tokens=20,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )

        generated = tokenizer.decode(
            output_ids[0, prompt_ids.shape[1]:],
            skip_special_tokens=True,
        ).strip()

        predicted = None
        cleaned = generated.lower().replace("*", "").replace(".", "").replace(",", "")
        for word in cleaned.split():
            if word in VALID_LOCATIONS:
                predicted = word
                break

        is_correct = (predicted == correct_answer)

        print(
            f"[{story_id}] type={item['type']} transfers={item['num_transfers']} | "
            f"correct={correct_answer} predicted={predicted} ({'✓' if is_correct else '✗'}) | "
            f"generated: {generated[:80]}"
        )

        # activations for last prompt token, all layers (skip embedding state at idx 0)
        hidden_states = outputs.hidden_states  # tuple: (emb, layer1, ..., layerN)
        layer_activations = torch.stack([h[0, -1, :] for h in hidden_states[1:]]).cpu()
        all_activations[story_id] = layer_activations

        results.append({
            "id": story_id,
            "type": item["type"],
            "num_transfers": item["num_transfers"],
            "num_location_distractors": item["num_location_distractors"],
            "num_random_distractors": item["num_random_distractors"],
            "object": item["object"],
            "correct_answer": correct_answer,
            "predicted": predicted,
            "generated_text": generated,
            "is_correct": is_correct,
            "prompt": tokenizer.decode(prompt_ids[0], skip_special_tokens=False),
        })

    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nBehavioral results saved to {results_file}")

    torch.save(all_activations, activations_file)
    print(f"Activations saved to {activations_file}")

    total = len(results)
    correct = sum(1 for r in results if r["is_correct"])
    no_prediction = sum(1 for r in results if r["predicted"] is None)

    print(f"\n── Summary for {model_name} ──")
    print(f"Total stories: {total}")
    print(f"Correct: {correct} ({100*correct/total:.1f}%)")
    print(f"No valid location predicted: {no_prediction} ({100*no_prediction/total:.1f}%)")

    for t in ["distractor", "red_herring", "reversal", "control"]:
        subset = [r for r in results if r["type"] == t]
        if subset:
            acc = sum(1 for r in subset if r["is_correct"]) / len(subset)
            print(f"  {t}: {acc*100:.1f}% ({len(subset)} stories)")

    print("\nAccuracy by num_transfers:")
    for n in sorted(set(r["num_transfers"] for r in results)):
        subset = [r for r in results if r["num_transfers"] == n]
        acc = sum(1 for r in subset if r["is_correct"]) / len(subset)
        print(f"  {n} transfers: {acc*100:.1f}% ({len(subset)} stories)")

if __name__ == "__main__":
    main()