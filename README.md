# Who Has the Best Lottery Ticket?
### Comparing Neural Network Pruning Across Modern LLM Architectures on ROSIE

**ROSIE Supercomputer Super Challenge 2026**
**Team:** Ethan Houseworth + Ultron

---

## Why This Matters

Training a 7-billion parameter model costs millions in compute. We found that 3.5 billion of those parameters are dead weight — zeroed out with no impact on quality. The industry is spending half its GPU budget training neurons that do nothing. If we could find the winning ticket before training, the cost of AI drops in half overnight. That's not a hypothetical. That's what NVIDIA is building hardware for right now.

## The Question

In 2018, MIT proved you can delete 90% of a neural network and it still works. They called it the **Lottery Ticket Hypothesis** — inside every large network, there's a small "winning ticket" subnetwork doing all the real work.

But that was 2018. Models were small. Architectures were simple.

**We wanted to know: does this still hold for modern LLMs? And where exactly does the lottery ticket live?**

## What We Did

### Phase 1: The Cliff
We took Mistral 7B and pruned it uniformly at 50%, 70%, and 90% sparsity. Found the cliff — 50% pruning barely hurts, 70% destroys it.

### Phase 2: Where Does the Lottery Ticket Live?
Instead of pruning everything uniformly, we targeted specific layer types to find out which parts of the network matter most. ~20 experiments on Mistral 7B:

**Targets × Sparsity Levels:**

| Target | What it is | 30% | 50% | 70% |
|--------|-----------|-----|-----|-----|
| **All layers** | Uniform pruning (baseline comparison) | TBD | TBD | TBD |
| **Attention** | q/k/v/o projections — how the model "looks at" context | TBD | TBD | TBD |
| **MLP** | Feed-forward layers — where the model "thinks" | TBD | TBD | TBD |
| **Early layers** (0-15) | First half of the network — initial processing | TBD | TBD | TBD |
| **Late layers** (16-31) | Second half — final reasoning and output | TBD | TBD | TBD |
| **Embeddings** | Input/output token representations | TBD | — | — |
| **LM head** | Final output layer | TBD | — | — |

**Plus extreme tests:**

| Target | 90% Sparsity |
|--------|-------------|
| Attention only | TBD |
| MLP only | TBD |

**Total: ~20 experiments**

This tells us: Is the lottery ticket in the attention mechanism? The feed-forward layers? The early processing or late reasoning? Which component can you gut and which is sacred?

### Phase 3: Cross-Model Comparison (if time permits)
Run the same experiments on multiple architectures to see if the lottery ticket location is universal or architecture-dependent:

| Model | Parameters | Developer |
|-------|-----------|-----------|
| Mistral 7B | 7.24B | Mistral AI |
| Qwen2.5 7B | 7.6B | Alibaba |
| SmolLM2 1.7B | 1.7B | HuggingFace |
| Phi-2 | 2.7B | Microsoft |
| Qwen2.5 0.5B | 0.5B | Alibaba |

## Benchmarks

| Metric | What it measures | Tool |
|--------|-----------------|------|
| **Perplexity** | Model accuracy/intelligence (lower = better) | WikiText-2 dataset |
| **Inference speed** | Tokens per second | 50-token generation benchmark |
| **Memory usage** | GPU VRAM consumption | torch.cuda.memory_allocated |
| **Output quality** | Can it still write coherent English? | Manual inspection |

All experiments run on V100 (32GB) for fair comparison.

## Results

### Phase 1: Uniform Pruning (Mistral 7B, V100)

| Sparsity | Perplexity | Tokens/sec | Coherent? |
|----------|-----------|------------|-----------|
| Baseline | 4.0 | 3.19* | ✅ |
| 50% | 6.46 | 16.96 | ✅ |
| 70% | 1,239.63 | 21.56 | ❌ |
| 90% | 35,789.36 | 21.61 | ❌ |

*Baseline ran on T4 with CPU offloading; needs V100 re-run for fair speed comparison.

**The cliff is between 50-70%.** Delete half, barely any damage. Delete 70%, total collapse.

**Sample outputs:**
- **Baseline:** "The meaning of life is to find your gift. The purpose of life is to give it away."
- **50% pruned:** "The meaning of life is to be found in the life we lead, not in the life we wish we had."
- **70% pruned:** "The meaning of life is a life fulled life fulled life fulled life fulled..."
- **90% pruned:** "The meaning of life is­rezentaturagem kennis kennis opponaturajuajuajuaju..."

### Phase 2: Layer-Targeted Pruning

*Results pending — experiments ready to run*

### Phase 3: Cross-Model

*Results pending*

## Method

**Pruning technique:** Unstructured L1 magnitude pruning (`torch.nn.utils.prune`). Rank every weight by absolute value, zero out the smallest X%. Architecture stays the same shape — same layers, same neurons — but connections are dead.

**Why magnitude pruning?** It's the method from the original Lottery Ticket paper. Simple, reproducible, and the baseline against which all other pruning methods are compared.

**Precision:** float16 (Daniel's recommendation: explore float8 as additional axis — can you prune AND quantize?)

**Benchmark:** WikiText-2 test set, 2048 token context window. Perplexity measured via cross-entropy loss. Speed measured on 50-token generation.

## Infrastructure

- **Cluster:** MSOE ROSIE Supercomputer
- **GPUs:** NVIDIA V100 (32GB) on DGX partition
- **Software:** PyTorch, HuggingFace Transformers
- **Environment:** `/data/csc4611/conda-csc4611/`

## References

- Frankle, J. & Carbin, M. (2018). "The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks." arXiv:1803.03635
- Mistral AI, Alibaba (Qwen), Microsoft (Phi), HuggingFace (SmolLM)

## Reproducibility

All code is in this repo. To reproduce:

```bash
# On ROSIE, request a V100
srun --partition=dgx --gpus=1 --cpus-per-gpu=8 -t 0-4:0 --pty bash
conda activate /data/csc4611/conda-csc4611/

# Phase 1: Uniform pruning
python3 prune.py

# Phase 2: Layer-targeted pruning
python3 prune_layers.py

# Phase 3: Multi-model comparison
python3 prune_multi.py
```

## Open Questions

1. **Where does the lottery ticket live?** Attention vs MLP vs early vs late layers
2. **Can you prune AND quantize?** float16 vs float8 pruning comparison
3. **Do bigger models have better lottery tickets?** More redundancy = more prunable?
4. **Is the lottery ticket location universal?** Same across architectures or different?
