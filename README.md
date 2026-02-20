# ROSIE Super Challenge 2026 — Lottery Ticket Pruning

**Competition:** ROSIE Supercomputer Super Challenge  
**Deadline:** Friday, March 27, 2026 @ 11:59 PM  
**Prize:** $10K / $6K / $4K + NVIDIA GPUs + DGX Spark  
**Judges:** Dwight Diercks (NVIDIA SVP), Nick Haemel (NVIDIA VP), Dr. Kedziora, Dr. Riley  
**Team:** Ethan Houseworth + Ultron  
**Idea from:** Daniel  
**Submit to:** Rosie_S.lqe6tdwasggor3ys@u.box.com (or riley@msoe.edu if file too large)

## Concept

**"How much of a neural network can you delete before it breaks?"**

Based on MIT's Lottery Ticket Hypothesis (Frankle & Carbin, 2018): inside any large neural network, there's a tiny subnetwork that does all the real work. We prove this on a modern 7B-parameter LLM using ROSIE's GPUs.

Take Mistral 7B, benchmark it, apply magnitude pruning at increasing levels (50/70/90%), benchmark again. Show the accuracy vs size tradeoff with clean visualizations.

## Model

**Mistral 7B v0.1** (`mistralai/Mistral-7B-v0.1`)
- 7.24B parameters
- Open weights, no auth required
- Well-known, strong baseline benchmarks
- Fits on T4 GPU (16GB VRAM) in float16

Originally planned LLaMA 3 8B but it's gated on HuggingFace. Mistral 7B is fully open and comparable.

## Benchmarks

| Metric | What it measures | Tool |
|--------|-----------------|------|
| **Perplexity** | Model accuracy/intelligence (lower = better) | WikiText-2 dataset |
| **Inference speed** | Tokens per second | Custom generation benchmark |
| **Memory usage** | GPU VRAM consumption | torch.cuda.memory_allocated |

**Story:** "We made the model X% smaller, Y% faster, and only lost Z% accuracy."

## Pruning Method

Unstructured magnitude pruning via `torch.nn.utils.prune`:
- Remove weights closest to zero (least important)
- Test at 50%, 70%, 90% sparsity levels
- Optional: fine-tune pruned model to recover accuracy

## Infrastructure

- **Cluster:** MSOE ROSIE (Ubuntu 24.04, SLURM)
- **GPU:** Tesla T4 (teaching partition)
- **Software:** PyTorch, HuggingFace Transformers, accelerate (all pre-installed)
- **Conda env:** `/data/csc4611/conda-csc4611/`
- **Working dir:** `~/pruning/`

## Timeline

- **Feb 20:** Environment confirmed, model loading, baseline benchmarks
- **Feb 21 - Mar 14:** Pruning experiments at 50/70/90%
- **Mar 15-26:** Package deliverable (poster/paper/video)
- **Mar 27:** Submit

## Results

### Baseline (Mistral 7B, unpruned, T4)
- Parameters: 7.24B
- Perplexity (WikiText-2): **4.0**
- Inference speed: 3.19 tokens/sec (CPU offloading due to T4 16GB limit)
- Memory: 13.76 GB

### Results on V100 (32GB VRAM, no offloading)

| Metric | Baseline* | 50% Pruned | 70% Pruned | 90% Pruned |
|--------|-----------|-----------|-----------|-----------|
| Perplexity | 4.0 | **6.46** | 1,239.63 | 35,789.36 |
| Tokens/sec | 3.19* | **16.96** | 21.56 | 21.61 |
| GPU mem | 13.76 GB | 14.48 GB | 14.89 GB | 14.89 GB |
| Layers pruned | 0 | 225 | 225 | 225 |
| Coherent? | ✅ | ✅ | ❌ | ❌ |

*Baseline ran on T4 with CPU offloading, speed not directly comparable.

### Key Findings
1. **50% pruning: the sweet spot.** Model stays coherent (perplexity 4.0 → 6.46), 5x faster inference.
2. **The cliff is between 50-70%.** At 70%, perplexity explodes 200x and output degenerates ("life fulled life fulled").
3. **90% = total collapse.** Gibberish output, perplexity 35,789.
4. **Speed gains plateau.** 50% → 90% pruning only goes from 16.96 → 21.61 tok/s, while accuracy is destroyed.

### Sample Outputs
- **Baseline:** "The meaning of life is to find your gift. The purpose of life is to give it away."
- **50% pruned:** "The meaning of life is to be found in the life we lead, not in the life we wish we had."
- **70% pruned:** "The meaning of life is a life fulled life fulled life fulled life fulled..."
- **90% pruned:** "The meaning of life is­rezentaturagem kennis kennis opponaturajuajuajuaju..."

## Status

- [x] ROSIE access confirmed (web shell, VPN on Mac mini)
- [x] GPU node allocated (T4 on dh-node3)
- [x] Conda env + transformers working
- [x] Model selected (Mistral 7B v0.1)
- [x] Baseline benchmarks (T4, perplexity 4.0, 3.19 tok/s)
- [x] 50% pruning (perplexity 6.46, 16.96 tok/s) ← SWEET SPOT
- [x] 70% pruning (perplexity 1239.63, model breaks)
- [x] 90% pruning (perplexity 35789.36, gibberish)
- [ ] Re-run baseline on V100 for fair speed comparison
- [ ] Deliverable created
- [ ] Submitted

## Notes

- SSH password auth not working — using web shell (OOD) for now. Need to email Dr. Riley for SSH reset.
- LLaMA 3 8B is gated, switched to Mistral 7B.
- Teaching partition T4 nodes are busy but available.
