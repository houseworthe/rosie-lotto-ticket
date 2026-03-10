# ROSIE Autoresearch — Small Model Fine-Tuning

**Goal:** Fine-tune small language models (Qwen3 0.6B–8B) on specific tasks to match or beat frontier LLMs (Opus, Sonnet, GPT-5), using Karpathy's autoresearch framework adapted for SLURM.

**Inspired by:** [Distil Labs](https://www.distilai.com/) results showing fine-tuned small models can beat GPT-4 on narrow tasks, and [Karpathy's autoresearch](https://github.com/karpathy/autoresearch) for autonomous experimentation.

## Concept

Instead of pretraining from scratch (original autoresearch), we **fine-tune** pretrained Qwen3 models on specific benchmarks. The autoresearch loop submits SLURM jobs to ROSIE, polls for completion, evaluates results, and iterates — all autonomously overnight.

**The bet:** A 0.6B–4B model, fine-tuned with the right data/hyperparameters, can match frontier models on narrow tasks. The agent explores the hyperparameter space while you sleep.

## Target Tasks

| Task | Dataset | Metric | Frontier Baseline |
|------|---------|--------|-------------------|
| **TREC Classification** | TREC-6 (question type, 6 coarse classes) | Accuracy | ~97% (GPT-4 zero-shot) |
| **Text2SQL** | Spider / WikiSQL | Execution accuracy | ~85% (GPT-4) |
| **E-commerce Classification** | Amazon product categorization | F1 score | ~95% (GPT-4) |

## Experiment Tracks

### Track A: Fine-Tuning (beat frontier on narrow tasks)
LoRA/QLoRA fine-tuning of Qwen3 0.6B–8B on specific benchmarks (TREC, Text2SQL, E-commerce). See target tasks above.

### Track B: Vocabulary Pruning (free VRAM savings)
Strip unused tokens from embedding/unembedding layers. Most models ship with vocabularies containing thousands of tokens never used in the target task (CJK characters, code tokens, etc.). Removing them saves ~1GB+ VRAM with **zero quality loss** — it's mathematically lossless for the task. Daniel already validated this on Qwen3 on his own hardware. This should be the **default first step** before any other optimization.

### Track C: Surgical Inter-Size Pruning
When a model family has size gaps (e.g. Qwen3-9B and Qwen3-27B), prune the larger model down to fill the gap (~15–20B). Use structured layer pruning (drop transformer blocks) and/or width pruning (reduce hidden dim). Find the sweet spot: significantly better than 9B quality at a fraction of 27B cost. Requires V100/H100 for the larger models.

### Track E: Non-English Token Pruning → Benchmark Validation
Strip non-English tokens (CJK, Arabic, Cyrillic, etc.) from the vocabulary for English-only tasks. Daniel's experiments show ~1GB VRAM savings, but the key question is: **does the model still perform identically on English benchmarks?** The model's internal representations might subtly depend on multilingual tokens. This track systematically measures the impact: prune increasing percentages of non-English vocab → eval on all target benchmarks → find the threshold where quality degrades (if it does).

### Track F: Vision Encoder Removal from VL Models
Qwen3-VL models have a vision encoder bolted onto the language backbone. Strip the vision components entirely and evaluate the remaining language model against the pure-language Qwen3 equivalent. Two interesting outcomes: (1) if it matches, you get a cheaper model from the VL weights, (2) if it's *better*, multimodal training improved language understanding — which is a publishable finding. Requires careful surgery to remove vision encoder + projection layers without breaking the language path.

### Track D: Combined Pipelines
The real magic is combining techniques. The agent should try different orderings and measure quality-per-VRAM:
- Vocab pruning → fine-tuning (baseline combo)
- Vocab pruning → layer pruning → fine-tuning
- Layer pruning → fine-tuning → vocab pruning
- Full pipeline: vocab prune → structured prune → LoRA fine-tune → quantize

Each experiment is a **5–10 minute SLURM job** so we can run hundreds per night (~6–12/hour × 8 hours = 50–100 experiments).

These were chosen because:
1. Well-defined, measurable outputs
2. Existing datasets with clear eval metrics
3. Distil Labs showed small models can win on similar tasks
4. Feasible to fine-tune on T4/V100 GPUs within SLURM time limits

## Architecture

```
┌─────────────────────────────────────┐
│  Agent (autoresearch loop)          │
│  - Reads program.md                 │
│  - Modifies finetune.py             │
│  - Submits SLURM jobs via SSH       │
│  - Polls for job completion         │
│  - Reads eval metrics from output   │
│  - Keeps/discards, iterates         │
└──────────┬──────────────────────────┘
           │ sbatch / squeue / cat slurm-*.out
           ▼
┌─────────────────────────────────────┐
│  ROSIE (SLURM cluster)             │
│  - teaching: T4 (16GB)             │
│  - dgx: V100 (32GB)               │
│  - dgxh100: H100 (80GB)           │
│                                     │
│  Each job:                          │
│  1. Loads Qwen3 model              │
│  2. Fine-tunes on task dataset     │
│  3. Evaluates on held-out test set │
│  4. Prints metrics to stdout       │
└─────────────────────────────────────┘
```

## Project Structure

```
README.md           — this file
program.md          — autoresearch agent instructions (adapted for SLURM)
finetune.py         — fine-tuning pipeline (agent-editable)
vocab_prune.py      — vocabulary pruning (agent-editable)
layer_prune.py      — structured layer/width pruning (agent-editable)
pipeline.py         — combined pipeline orchestrator (agent-editable)
prepare.py          — one-time dataset download and preprocessing
slurm_template.sh   — SBATCH job template
eval.py             — evaluation harness (fixed, agent cannot modify)
requirements.txt    — pip dependencies for ROSIE conda env
results.tsv         — experiment log (untracked)
```

## Quick Start

```bash
# 1. SSH into ROSIE
ssh houseworthe@dh-mgmt2.hpc.msoe.edu

# 2. Clone this repo
git clone https://github.com/houseworthe/rosie-lotto-ticket.git ~/autoresearch
cd ~/autoresearch

# 3. Set up conda environment
conda create -n autoresearch python=3.11 -y
conda activate autoresearch
pip install -r requirements.txt

# 4. Download datasets (run on management node, no GPU needed)
python prepare.py

# 5. Test a single fine-tuning run
sbatch slurm_template.sh

# 6. Once working, point your agent at program.md and let it go
```

## Models

| Model | Parameters | VRAM (bf16) | Fits on |
|-------|-----------|-------------|---------|
| Qwen3-0.6B | 0.6B | ~1.5GB | T4 ✅ |
| Qwen3-1.7B | 1.7B | ~4GB | T4 ✅ |
| Qwen3-4B | 4B | ~9GB | T4 ✅, V100 ✅ |
| Qwen3-8B | 8B | ~17GB | V100 ✅, H100 ✅ |

We use LoRA (Low-Rank Adaptation) to make fine-tuning feasible on smaller GPUs. Full fine-tuning is an option the agent can explore on V100/H100.

## Previous Work (Lottery Ticket)

This project was originally a Lottery Ticket Hypothesis experiment on Mistral 7B for the ROSIE Super Challenge 2026. Key finding: 50% magnitude pruning preserves coherence (perplexity 4.0→6.46) with 5x speedup, but 70%+ causes collapse. That work is preserved in git history.

## Links

- [Karpathy's autoresearch](https://github.com/karpathy/autoresearch)
- [Distil Labs](https://www.distilai.com/)
- [Qwen3 Models](https://huggingface.co/Qwen)
- [ROSIE User Guide](https://msoe.dev/)
