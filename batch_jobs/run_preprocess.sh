#!/usr/bin/env bash
# batch_jobs/run_preprocess.sh

set -e
echo "[$(date)] Running preprocess..."
python3 src/preprocess.py
echo "[$(date)] Preprocess complete."
