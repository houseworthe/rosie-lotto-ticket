#!/bin/bash
#SBATCH --job-name=autoresearch
#SBATCH --partition=teaching
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.out

# Autoresearch job for ROSIE
# Usage:
#   sbatch slurm_template.sh                                    # runs finetune.py
#   sbatch --export=SCRIPT=vocab_prune.py slurm_template.sh     # runs vocab_prune.py
#   sbatch --export=SCRIPT=layer_prune.py slurm_template.sh     # runs layer_prune.py
#   sbatch --export=SCRIPT=pipeline.py slurm_template.sh        # runs pipeline.py
#
# For larger models (8B+), use:
#   sbatch --partition=dgx slurm_template.sh

# Default script if not specified
SCRIPT=${SCRIPT:-finetune.py}

# Load required modules (conda module may not exist on all nodes)
module load conda 2>/dev/null || true
module load cuda 2>/dev/null || true

# Ensure conda is available
if ! command -v conda &>/dev/null; then
    source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || \
    source /opt/conda/etc/profile.d/conda.sh 2>/dev/null || \
    source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || \
    { echo "ERROR: conda not found"; exit 1; }
fi

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Script: $SCRIPT"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "Started: $(date)"
echo "=========================================="

echo ""
echo "--- nvidia-smi (start of job) ---"
nvidia-smi
echo "---"
echo ""

# Activate conda environment
conda activate autoresearch

echo "Python: $(python --version)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

export HF_HOME=~/autoresearch/.hf_cache
export TRANSFORMERS_CACHE=~/autoresearch/.hf_cache
export CUBLAS_WORKSPACE_CONFIG=:4096:8

cd ~/autoresearch
python $SCRIPT

echo ""
echo "--- nvidia-smi (end of job) ---"
nvidia-smi
echo "---"
echo ""
echo "=========================================="
echo "Finished: $(date)"
echo "=========================================="
