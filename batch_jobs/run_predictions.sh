#!/usr/bin/env bash
# batch_jobs/run_prediction.sh
set -e
echo "[$(date)] Running prediction..."
python3 src/predict.py
echo "[$(date)] Prediction complete."
