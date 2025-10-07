#!/usr/bin/env bash
# batch_jobs/run_all.sh

set -e
echo "========== PIPELINE START: $(date) =========="
bash "$(dirname "$0")/run_preprocess.sh"
bash "$(dirname "$0")/run_training.sh"
bash "$(dirname "$0")/run_predictions.sh"
echo "========== PIPELINE END: $(date) =========="
