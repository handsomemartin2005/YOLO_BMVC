#!/usr/bin/env bash
set -euo pipefail

python scripts/check_package.py
python visualize_dataset.py --split test --max-images 12
python train.py --epochs 300 --imgsz 1280 --batch 4 --workers 4 --device 0
python evaluate.py --split test --imgsz 1280 --batch 4 --workers 4 --device 0
python visualize_predictions.py --split test --imgsz 1280 --conf 0.25 --device 0
python visualize_innov_features.py --split test --index 0 --imgsz 1280 --device 0
