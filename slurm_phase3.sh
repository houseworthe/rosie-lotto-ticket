#!/bin/bash
#SBATCH --job-name=phase3-router
#SBATCH --partition=dgx
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/home/houseworthe/autoresearch/phase3_%j.out
#SBATCH --error=/home/houseworthe/autoresearch/phase3_%j.err

echo "=== Phase 3: Router Training ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $HOSTNAME"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "Start: $(date)"
echo ""

# Conda setup for dgx nodes
source /usr/local/miniforge/miniforge3/etc/profile.d/conda.sh
conda activate autoresearch

cd ~/autoresearch

# Run router training
python -u router.py \
    --epochs 5 \
    --batch-size 4 \
    --lr 1e-3 \
    --top-k 2 \
    --max-train-samples 2000 \
    --max-eval-samples 500 \
    --sparsity-lambda 0.01 \
    --balance-lambda 0.01

echo ""
echo "=== Done: $(date) ==="
