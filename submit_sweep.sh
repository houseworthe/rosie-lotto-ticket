#!/bin/bash
# Submit all remaining TREC-6 sweep experiments in parallel
# Each job uses env vars to override finetune.py defaults
# r32 is already running as a separate job

echo "Submitting TREC-6 sweep experiments to dgx partition..."
echo ""

# LR sweep (all with default r16/alpha32)
echo "1. LR=1e-4 (half baseline):"
sbatch -p dgx --export=ALL,AR_LR=0.0001 slurm_template.sh

echo "2. LR=5e-4 (2.5x baseline):"
sbatch -p dgx --export=ALL,AR_LR=0.0005 slurm_template.sh

echo "3. LR=5e-5 (10x lower):"
sbatch -p dgx --export=ALL,AR_LR=0.00005 slurm_template.sh

# Rank sweep
echo "4. LoRA r64, alpha=128:"
sbatch -p dgx --export=ALL,AR_LORA_RANK=64,AR_LORA_ALPHA=128 slurm_template.sh

# More epochs
echo "5. 5 epochs (r16 baseline config):"
sbatch -p dgx --export=ALL,AR_EPOCHS=5 slurm_template.sh

# Scale up models
echo "6. Qwen3-1.7B:"
sbatch -p dgx --export=ALL,AR_MODEL=Qwen/Qwen3-1.7B slurm_template.sh

echo "7. Qwen3-4B:"
sbatch -p dgx --export=ALL,AR_MODEL=Qwen/Qwen3-4B slurm_template.sh

echo ""
echo "All jobs submitted. Run 'squeue -u \$USER' to check status."
