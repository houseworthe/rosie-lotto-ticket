#!/bin/bash
# Deploy Track D fix and submit both jobs

cd ~/autoresearch || exit 1

# Pull latest code (includes bug fix)
echo "Pulling latest code..."
git pull

# Submit Track D v5 (with eval bug fix)
echo "Submitting Track D v5 (pipeline)..."
export SCRIPT=pipeline.py
sbatch -p dgx slurm_template.sh

# Submit Text2SQL with 2hr time limit
echo "Submitting Text2SQL (2hr limit)..."
export AR_TASK=text2sql
sbatch -p dgx --time=02:00:00 slurm_template.sh

# Show queue status
echo ""
echo "Job queue:"
squeue -u houseworthe
