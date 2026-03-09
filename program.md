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
- **Fine-tuning, not pretraining**: We load pretrained Qwen3 models and adapt with LoRA
- **Multiple tasks**: Each experiment targets a specific task (TREC, Text2SQL, E-commerce)
- **Fixed time budget**: 30 minutes per SLURM job (not 5 min like original)

## What You CAN Do

- Modify `finetune.py` — this is your main file. Everything is fair game:
  - LoRA rank, alpha, target modules
  - Learning rate, batch size, warmup steps
  - Base model selection (Qwen3-0.6B through 8B)
  - Training data sampling/augmentation strategy
  - Prompt format and template
  - Number of epochs
- Modify `slurm_template.sh` — adjust partition, GPU count, time limit
- Switch between tasks by changing the `TASK` variable in finetune.py

## What You CANNOT Do

- Modify `prepare.py` or `eval.py` — these are read-only
- Install new packages beyond what's in `requirements.txt`
- Modify the evaluation metrics — `eval.py` is ground truth

## The Goal

For each task, get the **highest eval metric** (accuracy for TREC, execution accuracy for Text2SQL, F1 for E-commerce). The target is to match or exceed frontier LLM baselines.

## Experimentation

Each experiment is a SLURM job that takes ~30 minutes. The workflow:

### Submitting a Job

```bash
# 1. Edit finetune.py with your experimental changes
# 2. Commit the change
git add finetune.py && git commit -m "experiment: <description>"

# 3. Submit to SLURM
sbatch slurm_template.sh
# Submitted batch job 12345

# 4. Poll for completion (check every 2-3 minutes)
squeue -u houseworthe
# When the job disappears from the queue, it's done

# 5. Read results
cat slurm-12345.out | grep "^eval_"
```

### Output Format

The finetune script prints a summary like:

```
---
task:             trec
model:            Qwen/Qwen3-0.6B
eval_accuracy:    0.9420
eval_f1:          0.9380
training_seconds: 1200.5
total_seconds:    1350.2
peak_vram_mb:     12500.0
lora_rank:        16
lora_alpha:       32
learning_rate:    2e-4
num_epochs:       3
num_train_samples: 5452
```

Extract the key metric: `grep "^eval_" slurm-<jobid>.out`

## Logging Results

Log to `results.tsv` (tab-separated):

```
commit	task	model	eval_metric	metric_value	status	description
```

Example:
```
commit	task	model	eval_metric	metric_value	status	description
a1b2c3d	trec	Qwen3-0.6B	accuracy	0.9420	keep	baseline LoRA r=16
b2c3d4e	trec	Qwen3-0.6B	accuracy	0.9580	keep	increase LoRA r=64
c3d4e5f	trec	Qwen3-1.7B	accuracy	0.9650	keep	upgrade to 1.7B model
d4e5f6g	text2sql	Qwen3-4B	exec_accuracy	0.0000	crash	OOM on T4
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

**Phase 1: Establish baselines**
- Run each task with Qwen3-0.6B, default LoRA (r=16), default hyperparams
- Run with Qwen3-1.7B for comparison

**Phase 2: Hyperparameter search**
- LoRA rank: try 8, 16, 32, 64, 128
- Learning rate: try 1e-5, 5e-5, 1e-4, 2e-4, 5e-4
- Batch size vs gradient accumulation tradeoffs
- Number of epochs: 1, 3, 5, 10

**Phase 3: Advanced techniques**
- Full fine-tuning (on V100/H100 for smaller models)
- QLoRA (4-bit quantization + LoRA)
- Different target modules (q_proj, k_proj, v_proj, o_proj, gate_proj, etc.)
- Prompt engineering: different instruction formats
- Data augmentation: paraphrasing, few-shot examples in training data

**Phase 4: Push boundaries**
- Qwen3-4B and 8B models
- Ensemble approaches
- Task-specific architectural tweaks
- Curriculum learning (easy → hard examples)

## SLURM Tips

- **Partition selection**: Start with `teaching` (T4). Use `dgx` (V100) for 8B models.
- **Job monitoring**: `squeue -u houseworthe` shows running/pending jobs
- **Cancel job**: `scancel <jobid>`
- **Max wall time**: 24 hours (but aim for 30min per experiment)
- **Queue wait**: If teaching is full, try `dgx` partition

## NEVER STOP

Once the loop begins, do NOT pause to ask the human. They may be asleep. You are autonomous. If you run out of ideas, think harder — try combining approaches, revisit near-misses, try more radical changes. The loop runs until manually stopped.
