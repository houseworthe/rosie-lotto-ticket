# ROSIE Autoresearch — Small Model Fine-Tuning

**Goal:** Fine-tune small language models (Qwen3 0.6B–8B) on specific tasks to match or beat frontier LLMs (Opus, Sonnet, GPT-5), using Karpathy's autoresearch framework adapted for SLURM.

**Inspired by:** [Distil Labs](https://www.distilai.com/) results showing fine-tuned small models can beat GPT-4 on narrow tasks, and [Karpathy's autoresearch](https://github.com/karpathy/autoresearch) for autonomous experimentation.

## Concept

Instead of pretraining from scratch (original autoresearch), we **fine-tune** pretrained Qwen3 models on specific benchmarks. The autoresearch loop submits SLURM jobs to ROSIE, polls for completion, evaluates results, and iterates — all autonomously overnight.

**The bet:** A 0.6B–4B model, fine-tuned with the right data/hyperparameters, can match frontier models on narrow tasks. The agent explores the hyperparameter space while you sleep.

## Target Tasks

| Task | Dataset | Metric | Frontier Baseline |
|------|---------|--------|-------------------|
| **TREC Classification** | TREC-50 (question type classification) | Accuracy | ~97% (GPT-4) |
| **Text2SQL** | Spider / WikiSQL | Execution accuracy | ~85% (GPT-4) |
| **E-commerce Classification** | Amazon product categorization | F1 score | ~95% (GPT-4) |

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
finetune.py         — the file the agent modifies (model, LoRA config, hyperparams)
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
