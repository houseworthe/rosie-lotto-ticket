import torch, time, json
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1", torch_dtype=torch.float16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
print("Model loaded")
print("Params:", sum(p.numel() for p in model.parameters())/1e9, "B")
print("GPU mem:", round(torch.cuda.memory_allocated()/1e9, 2), "GB")

print("Loading WikiText-2...")
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
text = "\n\n".join(dataset["text"])
encodings = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
input_ids = encodings.input_ids.to("cuda")
with torch.no_grad():
    outputs = model(input_ids, labels=input_ids)
perplexity = torch.exp(outputs.loss).item()
print("Perplexity:", round(perplexity, 2))

print("Speed test...")
inp = tokenizer("The meaning of life is", return_tensors="pt").to("cuda")
s = time.time()
out = model.generate(**inp, max_new_tokens=50, pad_token_id=tokenizer.eos_token_id)
elapsed = time.time() - s
tps = 50/elapsed
print("Tokens/sec:", round(tps, 2))
print("Output:", tokenizer.decode(out[0], skip_special_tokens=True))

results = {"params_b": 7.24, "gpu_mem_gb": round(torch.cuda.memory_allocated()/1e9, 2), "perplexity": round(perplexity, 2), "tokens_per_sec": round(tps, 2)}
print("RESULTS:", json.dumps(results))
