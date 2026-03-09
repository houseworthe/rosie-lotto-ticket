#!/bin/bash
#SBATCH --job-name=autoresearch
#SBATCH --partition=teaching
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --time=00:15:00
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

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Script: $SCRIPT"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "Started: $(date)"
echo "=========================================="

bash --login -c "
    conda activate autoresearch 2>/dev/null || conda activate /data/csc4611/conda-csc4611/

    echo 'Python: '\$(python --version)
    echo 'PyTorch: '\$(python -c 'import torch; print(torch.__version__)')
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    echo ''

    export HF_HOME=~/autoresearch/.hf_cache
    export TRANSFORMERS_CACHE=~/autoresearch/.hf_cache

    cd ~/autoresearch
    python $SCRIPT
"

echo ""
echo "=========================================="
echo "Finished: $(date)"
echo "=========================================="
