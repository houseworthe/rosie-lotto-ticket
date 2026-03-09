#!/bin/bash
#SBATCH --job-name=autoresearch
#SBATCH --partition=teaching
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.out

# Autoresearch fine-tuning job for ROSIE
# Usage: sbatch slurm_template.sh
#
# For larger models (8B), use:
#   #SBATCH --partition=dgx
#   #SBATCH --gpus=1

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "Started: $(date)"
echo "=========================================="

# Activate conda environment
bash --login -c "
    conda activate autoresearch 2>/dev/null || conda activate /data/csc4611/conda-csc4611/
    
    # Show environment info
    echo 'Python: '$(python --version)
    echo 'PyTorch: '$(python -c 'import torch; print(torch.__version__)')
    echo 'CUDA: '$(python -c 'import torch; print(torch.version.cuda)')
    nvidia-smi
    echo ''
    
    # Set HF cache to shared storage to avoid re-downloading models
    export HF_HOME=~/autoresearch/.hf_cache
    export TRANSFORMERS_CACHE=~/autoresearch/.hf_cache
    
    # Run fine-tuning
    cd ~/autoresearch
    python finetune.py
"

echo ""
echo "=========================================="
echo "Finished: $(date)"
echo "=========================================="
