import torch, time, json
import torch.nn.utils.prune as prune
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

MODELS = [
    ("mistralai/Mistral-7B-v0.1", "Mistral-7B"),
    ("Qwen/Qwen2.5-7B", "Qwen2.5-7B"),
    ("HuggingFaceTB/SmolLM2-1.7B", "SmolLM2-1.7B"),
    ("microsoft/phi-2", "Phi-2"),
    ("Qwen/Qwen2.5-0.5B", "Qwen2.5-0.5B"),
]

SPARSITY_LEVELS = [0.0, 0.5, 0.7, 0.9]

print("Loading WikiText-2...")
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
text = "\n\n".join(dataset["text"])

all_results = []

for model_id, model_name in MODELS:
    print(f"\n{'='*60}")
    print(f"MODEL: {model_name} ({model_id})")
    print(f"{'='*60}")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    encodings = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)

    for sparsity in SPARSITY_LEVELS:
        label = "baseline" if sparsity == 0.0 else f"{int(sparsity*100)}%"
        print(f"\n--- {model_name} @ {label} ---")

        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")
        params = sum(p.numel() for p in model.parameters()) / 1e9

        if sparsity > 0:
            pruned_count = 0
            for name, module in model.named_modules():
                if isinstance(module, torch.nn.Linear):
                    if module.weight.device.type != "meta":
                        prune.l1_unstructured(module, name="weight", amount=sparsity)
                        prune.remove(module, "weight")
                        pruned_count += 1
            print(f"Pruned {pruned_count} layers at {int(sparsity*100)}%")

        gpu_mem = round(torch.cuda.memory_allocated() / 1e9, 2)
        print(f"Params: {params:.2f}B | GPU mem: {gpu_mem} GB")

        input_ids = encodings.input_ids.to("cuda")
        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
        ppl = torch.exp(outputs.loss).item()
        print(f"Perplexity: {round(ppl, 2)}")

        inp = tokenizer("The meaning of life is", return_tensors="pt").to("cuda")
        s = time.time()
        out = model.generate(**inp, max_new_tokens=50, pad_token_id=tokenizer.eos_token_id)
        tps = 50 / (time.time() - s)
        output_text = tokenizer.decode(out[0], skip_special_tokens=True)
        print(f"Tokens/sec: {round(tps, 2)}")
        print(f"Output: {output_text[:200]}")

        result = {
            "model": model_name,
            "params_b": round(params, 2),
            "sparsity": label,
            "perplexity": round(ppl, 2),
            "tokens_per_sec": round(tps, 2),
            "gpu_mem_gb": gpu_mem,
            "output_sample": output_text[:200]
        }
        all_results.append(result)

        del model
        torch.cuda.empty_cache()

print(f"\n{'='*60}")
print("ALL RESULTS")
print(f"{'='*60}")
for r in all_results:
    print(json.dumps(r))

with open("results.json", "w") as f:
    json.dump(all_results, f, indent=2)
print("\nResults saved to results.json")
print("DONE")
