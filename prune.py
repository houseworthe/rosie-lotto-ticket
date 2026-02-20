import torch, time, json
import torch.nn.utils.prune as prune
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
text = "\n\n".join(dataset["text"])
encodings = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
for sparsity in [0.5, 0.7, 0.9]:
    print(f"\n=== {int(sparsity*100)}% PRUNING ===")
    model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1", torch_dtype=torch.float16, device_map="auto")
    pruned_count = 0
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            if module.weight.device.type != "meta":
                prune.l1_unstructured(module, name="weight", amount=sparsity)
                prune.remove(module, "weight")
                pruned_count += 1
    print(f"Pruned {pruned_count} layers at {int(sparsity*100)}%")
    print(f"GPU mem: {round(torch.cuda.memory_allocated()/1e9, 2)} GB")
    input_ids = encodings.input_ids.to("cuda")
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
    ppl = torch.exp(outputs.loss).item()
    print(f"Perplexity: {round(ppl, 2)}")
    inp = tokenizer("The meaning of life is", return_tensors="pt").to("cuda")
    s = time.time()
    out = model.generate(**inp, max_new_tokens=50, pad_token_id=tokenizer.eos_token_id)
    tps = 50/(time.time()-s)
    print(f"Tokens/sec: {round(tps, 2)}")
    print(f"Output: {tokenizer.decode(out[0], skip_special_tokens=True)}")
    del model
    torch.cuda.empty_cache()
print("\nDONE")
