#!/usr/bin/env bash
set -euo pipefail

CONFIG=${1:-config.yaml}

echo "[1/3] Training DGANM with the configured evaluation protocol"
python train.py --config "$CONFIG"

echo "[2/3] Evaluating saved predictions and metrics"
python evaluate.py --config "$CONFIG"

echo "[3/3] Regenerating high-resolution figures"
python make_figures.py --config "$CONFIG"

echo "Completed. Results are available in outputs/."
