#!/bin/bash
# Package autoresearch results for Daniel
# Excludes raw data (37GB) and git history
cd ~/autoresearch

echo "=== Packaging for Daniel ==="
echo "Including: checkpoints, output, source, SLURM logs, results"

tar czf ~/autoresearch-daniel.tar.gz \
  --exclude='data' \
  --exclude='.git' \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  --exclude='wandb' \
  checkpoints/ \
  output/ \
  *.py \
  *.sh \
  *.md \
  *.out \
  *.tsv \
  *.txt \
  *.json \
  2>/dev/null

echo ""
echo "=== Package created ==="
ls -lh ~/autoresearch-daniel.tar.gz
echo ""
echo "=== Contents ==="
tar tzf ~/autoresearch-daniel.tar.gz | head -40
echo "... (truncated)"
echo ""
echo "Total files:"
tar tzf ~/autoresearch-daniel.tar.gz | wc -l
