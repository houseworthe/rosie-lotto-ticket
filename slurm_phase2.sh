#!/bin/bash
#SBATCH --job-name=phase2-multi-adapter
#SBATCH --partition=dgx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=phase2_%j.out
#SBATCH --error=phase2_%j.err

# Phase 2: Multi-Adapter Loading & Composition Testing
# Partition: dgx (V100) — DO NOT USE T4 (cuBLAS crash)

echo "=== Phase 2: Multi-Adapter Loading ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Started: $(date)"
echo ""

# Conda setup (compute nodes need direct source)
source /usr/local/miniforge/miniforge3/etc/profile.d/conda.sh
conda activate autoresearch

cd ~/autoresearch

# Run phase 2 (all tests: verify + blend + cross-eval)
python multi_adapter.py 2>&1

echo ""
echo "=== Phase 2 Complete ==="
echo "Finished: $(date)"
