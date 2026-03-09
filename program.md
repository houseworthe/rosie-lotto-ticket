# autoresearch — ROSIE Fine-Tuning

This is an experiment to have an LLM autonomously fine-tune small models to beat frontier LLMs on specific tasks, using ROSIE's SLURM cluster.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar9`). The branch `autoresearch/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current main.
3. **Read the in-scope files**:
   - `README.md` — project context and goals
   - `prepare.py` — dataset download and preprocessing (do not modify)
   - `eval.py` — evaluation harness (do not modify)
   - `finetune.py` — the file you modify. LoRA config, hyperparameters, training loop.
   - `slurm_template.sh` — SLURM job template (you may adjust resource requests)
4. **Verify datasets exist**: Check that `~/autoresearch/data/` contains task datasets. If not, tell the human to run `python prepare.py` on ROSIE.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row.
6. **Confirm and go**: Confirm setup looks good.

## Key Differences from Original Autoresearch

- **SLURM, not local GPU**: You submit jobs via `sbatch`, not `uv run train.py`
- **Async execution**: After submitting, poll with `squeue -u houseworthe` until done, then read `slurm-<jobid>.out`
- **Multiple tracks**: Fine-tuning, vocab pruning, layer pruning, and combined pipelines
- **Multiple tasks**: Each experiment targets a specific task (TREC, Text2SQL, E-commerce)
- **Fixed time budget**: 5–10 minutes per SLURM job so you can run hundreds overnight

## Experiment Tracks

You have FOUR tracks to explore. Mix and match across them:

### Track A: Fine-Tuning (`finetune.py`)
LoRA/QLoRA fine-tuning of Qwen3 0.6B–8B on task benchmarks. Edit hyperparams, LoRA config, model size, prompt templates.

### Track B: Vocabulary Pruning (`vocab_prune.py`)
Strip unused tokens from embedding/unembedding layers. **Lossless** — zero quality impact, free VRAM savings. This should be the **default first step** before any other optimization. Experiment with pruning strategies (task-frequency, charset, hybrid).

### Track C: Surgical Layer Pruning (`layer_prune.py`)
Drop transformer layers from larger models to fill size gaps in the model family. E.g. prune Qwen3-27B → ~15-20B. Try different strategies: uniform, importance-based, middle-out. Use V100/H100 for larger models.

### Track D: Combined Pipelines (`pipeline.py`)
Run multiple techniques in sequence. The real value is finding the best ordering:
- Vocab prune → fine-tune (baseline combo)
- Vocab prune → layer prune → fine-tune
- Layer prune → fine-tune → vocab prune
- Full pipeline: vocab prune → layer prune → LoRA → quantize

**Priority**: Start with Track B (vocab pruning — quick wins, validates the setup), then Track A (fine-tuning baselines), then Track D (combinations), then Track C (needs larger GPUs).

## What You CAN Do

- Modify `finetune.py` — LoRA config, hyperparams, model selection, prompt templates
- Modify `vocab_prune.py` — pruning strategy, charset rules, frequency thresholds
- Modify `layer_prune.py` — drop strategy, fraction, width pruning
- Modify `pipeline.py` — pipeline ordering, step combinations
- Modify `slurm_template.sh` — partition, GPU count, time limit, which script to run
- Switch between tasks by changing the `TASK` variable in any script

## What You CANNOT Do

- Modify `prepare.py` or `eval.py` — these are read-only
- Install new packages beyond what's in `requirements.txt`
- Modify the evaluation metrics — `eval.py` is ground truth

## The Goal

Maximize **quality per VRAM** — the best eval metric at the lowest memory footprint. For each task, match or exceed frontier LLM baselines while minimizing model size. Track all results across all four tracks.

## Experimentation

Each experiment is a SLURM job that takes ~30 minutes. The workflow:

### Submitting a Job

```bash
# 1. Edit the target script (finetune.py, vocab_prune.py, layer_prune.py, or pipeline.py)
# 2. Commit the change
git add -A && git commit -m "experiment: <description>"

# 3. Submit to SLURM (set SCRIPT env var for non-default scripts)
sbatch slurm_template.sh                          # runs finetune.py (default)
sbatch --export=SCRIPT=vocab_prune.py slurm_template.sh   # runs vocab_prune.py
sbatch --export=SCRIPT=layer_prune.py slurm_template.sh   # runs layer_prune.py
sbatch --export=SCRIPT=pipeline.py slurm_template.sh      # runs pipeline.py

# 4. Poll for completion (check every 1-2 minutes — jobs are 5-10 min)
squeue -u houseworthe
# When the job disappears from the queue, it's done

# 5. Read results
grep "^track:\|^eval_\|^task:\|^model:\|^param_reduction:\|^mb_saved:\|^peak_vram" slurm-<jobid>.out
```

### Output Format

All scripts print a `---` delimited summary block. Key fields:

```
---
track:            <finetune|vocab_prune|layer_prune|pipeline>
task:             trec
model:            Qwen/Qwen3-0.6B
eval_accuracy:    0.9420
param_reduction:  35.2%
peak_vram_mb:     8500.0
total_seconds:    420.1
```

Extract metrics: `grep "^track:\|^eval_\|^param_\|^peak_vram\|^mb_saved" slurm-<jobid>.out`

## Logging Results

Log to `results.tsv` (tab-separated):

```
commit	track	task	model	eval_metric	metric_value	peak_vram_mb	status	description
```

Example:
```
commit	track	task	model	eval_metric	metric_value	peak_vram_mb	status	description
a1b2c3d	vocab_prune	trec	Qwen3-0.6B	accuracy	0.9400	800.0	keep	baseline vocab prune (task_frequency)
b2c3d4e	finetune	trec	Qwen3-0.6B	accuracy	0.9420	4500.0	keep	baseline LoRA r=16
c3d4e5f	pipeline	trec	Qwen3-0.6B	accuracy	0.9580	4200.0	keep	vocab_prune -> finetune
d4e5f6g	layer_prune	trec	Qwen3-4B	accuracy	0.8900	6000.0	keep	drop 25% layers (importance)
e5f6g7h	pipeline	trec	Qwen3-4B	accuracy	0.9650	5500.0	keep	vocab + layer prune + finetune
f6g7h8i	finetune	text2sql	Qwen3-8B	exec_accuracy	0.0000	0.0	crash	OOM on T4
```

## The Experiment Loop

LOOP FOREVER:

1. Look at `results.tsv` — what's been tried, what worked
2. Choose an experiment: modify `finetune.py` (and optionally `slurm_template.sh`)
3. `git commit` the changes
4. Submit: `sbatch slurm_template.sh`
5. Poll: `squeue -u houseworthe` every 2-3 minutes until job completes
6. Read results: `grep "^eval_\|^task:\|^model:" slurm-<jobid>.out`
7. If empty/crash: `tail -n 50 slurm-<jobid>.out` for error diagnosis
8. Log to `results.tsv`
9. If improved → keep commit, advance branch
10. If worse → `git reset --hard HEAD~1`, try something else

### Experiment Ideas (in rough priority order)

**Phase 1: Quick wins — Vocab Pruning (Track B)**
- Vocab prune Qwen3-0.6B for TREC → measure VRAM savings, confirm zero quality loss
- Try all three strategies: task_frequency, charset, hybrid
- Vocab prune Qwen3-1.7B, 4B → bigger models = bigger absolute savings
- This validates the setup fast (runs in <2 min)

**Phase 2: Fine-tuning baselines (Track A)**
- Run each task with Qwen3-0.6B, default LoRA (r=16)
- Run with Qwen3-1.7B for comparison
- LoRA rank sweep: 8, 16, 32, 64
- Learning rate sweep: 1e-5, 5e-5, 1e-4, 2e-4, 5e-4

**Phase 3: Combined pipelines (Track D)**
- Vocab prune → LoRA fine-tune (the obvious combo)
- Compare ordering: fine-tune first vs prune first
- Vocab prune → layer prune → fine-tune
- Full pipeline with quantization

**Phase 4: Surgical pruning (Track C)**
- Layer prune Qwen3-4B by 25% → eval quality retention
- Try all strategies: uniform, last, middle, importance
- If V100 available: prune Qwen3-27B → ~15-20B, compare vs Qwen3-9B

**Phase 5: Push boundaries**
- QLoRA (4-bit + LoRA) for 8B models on T4
- Full fine-tuning (V100/H100 for smaller models)
- Data augmentation, prompt engineering
- Best pipeline per task: find the optimal combo for TREC, Text2SQL, E-commerce

## SLURM Tips

- **Partition selection**: Start with `teaching` (T4). Use `dgx` (V100) for 8B models.
- **Job monitoring**: `squeue -u houseworthe` shows running/pending jobs
- **Cancel job**: `scancel <jobid>`
- **Max wall time**: 24 hours (but aim for 30min per experiment)
- **Queue wait**: If teaching is full, try `dgx` partition

## NEVER STOP

Once the loop begins, do NOT pause to ask the human. They may be asleep. You are autonomous. If you run out of ideas, think harder — try combining approaches, revisit near-misses, try more radical changes. The loop runs until manually stopped.
