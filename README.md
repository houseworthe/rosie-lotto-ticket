# Who Has the Best Lottery Ticket?
### Comparing Neural Network Pruning Across Modern LLM Architectures on ROSIE

**ROSIE Supercomputer Super Challenge 2026**
**Team:** Ethan Houseworth + Ultron

---

## The Question

In 2018, MIT proved you can delete 90% of a neural network and it still works. They called it the **Lottery Ticket Hypothesis** — inside every large network, there's a small "winning ticket" subnetwork doing all the real work.

But that was 2018. Models were small. Architectures were simple.

**We wanted to know: does this still hold for modern LLMs? And which architectures have the best lottery tickets?**

## What We Did

We took 5 open-source language models ranging from 0.5B to 7.6B parameters, ran them on ROSIE's GPUs, and systematically pruned them at 50%, 70%, and 90% sparsity using magnitude pruning. At each level, we measured:

- **Perplexity** (WikiText-2) — how smart is the model? Lower = better
- **Inference speed** — tokens per second
- **Memory usage** — GPU VRAM consumption
- **Output quality** — can it still write coherent English?

## The Models

| Model | Parameters | Developer | Why We Picked It |
|-------|-----------|-----------|-----------------|
| **Mistral 7B** | 7.24B | Mistral AI | Top open 7B model, strong baseline |
| **Qwen2.5 7B** | 7.6B | Alibaba | Competes with Mistral, different architecture |
| **SmolLM2 1.7B** | 1.7B | HuggingFace | Purpose-built small model |
| **Phi-2** | 2.7B | Microsoft | Punches above its weight class |
| **Qwen2.5 0.5B** | 0.5B | Alibaba | Can a sub-1B model survive pruning at all? |

## Results

### Mistral 7B (7.24B params)

| Sparsity | Perplexity | Tokens/sec | GPU Memory | Coherent? |
|----------|-----------|------------|------------|-----------|
| Baseline | 4.0 | 3.19* | 13.76 GB | ✅ |
| 50% | 6.46 | 16.96 | 14.48 GB | ✅ |
| 70% | 1,239.63 | 21.56 | 14.89 GB | ❌ |
| 90% | 35,789.36 | 21.61 | 14.89 GB | ❌ |

*Baseline ran on T4 with CPU offloading; pruned versions ran on V100.

**Finding:** The cliff is between 50-70%. Half the network deleted, barely any damage. Delete 70% and it falls apart.

### Qwen2.5 7B (7.6B params)

| Sparsity | Perplexity | Tokens/sec | GPU Memory | Coherent? |
|----------|-----------|------------|------------|-----------|
| Baseline | TBD | TBD | TBD | TBD |
| 50% | TBD | TBD | TBD | TBD |
| 70% | TBD | TBD | TBD | TBD |
| 90% | TBD | TBD | TBD | TBD |

### SmolLM2 1.7B (1.7B params)

| Sparsity | Perplexity | Tokens/sec | GPU Memory | Coherent? |
|----------|-----------|------------|------------|-----------|
| Baseline | TBD | TBD | TBD | TBD |
| 50% | TBD | TBD | TBD | TBD |
| 70% | TBD | TBD | TBD | TBD |
| 90% | TBD | TBD | TBD | TBD |

### Phi-2 (2.7B params)

| Sparsity | Perplexity | Tokens/sec | GPU Memory | Coherent? |
|----------|-----------|------------|------------|-----------|
| Baseline | TBD | TBD | TBD | TBD |
| 50% | TBD | TBD | TBD | TBD |
| 70% | TBD | TBD | TBD | TBD |
| 90% | TBD | TBD | TBD | TBD |

### Qwen2.5 0.5B (0.5B params)

| Sparsity | Perplexity | Tokens/sec | GPU Memory | Coherent? |
|----------|-----------|------------|------------|-----------|
| Baseline | TBD | TBD | TBD | TBD |
| 50% | TBD | TBD | TBD | TBD |
| 70% | TBD | TBD | TBD | TBD |
| 90% | TBD | TBD | TBD | TBD |

## Key Findings

1. **TBD** — Which model has the best lottery ticket?
2. **TBD** — Do bigger models prune better than small ones?
3. **TBD** — Where is the universal "cliff" where models break?
4. **TBD** — Is there an architecture that handles pruning gracefully?

## Sample Outputs at 90% Pruning

| Model | Output |
|-------|--------|
| Mistral 7B | "­rezentaturagem kennis kennis opponaturajuajuaju..." |
| Qwen2.5 7B | TBD |
| SmolLM2 1.7B | TBD |
| Phi-2 | TBD |
| Qwen2.5 0.5B | TBD |

## Method

**Pruning technique:** Unstructured L1 magnitude pruning (`torch.nn.utils.prune`). We rank every weight by absolute value and zero out the smallest X%. The architecture stays the same shape — same layers, same neurons — but most connections are dead.

**Why magnitude pruning?** It's the method from the original Lottery Ticket paper. Simple, reproducible, and the baseline against which all other pruning methods are compared.

**Benchmark:** WikiText-2 test set, 2048 token context window. Perplexity measured via cross-entropy loss. Speed measured on 50-token generation.

## Infrastructure

- **Cluster:** MSOE ROSIE Supercomputer
- **GPUs:** NVIDIA Tesla T4 (16GB), V100 (32GB)
- **Software:** PyTorch, HuggingFace Transformers
- **Environment:** `/data/csc4611/conda-csc4611/`

## References

- Frankle, J. & Carlin, M. (2018). "The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks." arXiv:1803.03635
- Mistral AI, Alibaba (Qwen), Microsoft (Phi), HuggingFace (SmolLM)

## Reproducibility

All code is in this repo. To reproduce:

```bash
# On ROSIE, request a V100
srun --partition=dgx --gpus=1 --cpus-per-gpu=8 -t 0-4:0 --pty bash
conda activate /data/csc4611/conda-csc4611/

# Run single model baseline + pruning
python3 prune.py

# Run all models
python3 prune_multi.py
```
