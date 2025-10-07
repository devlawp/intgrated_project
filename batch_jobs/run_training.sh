#!/usr/bin/env bash
# batch_jobs/run_training.sh
set -e
echo "[$(date)] Running training..."
python3 src/train_model.py
echo "[$(date)] Training complete."
