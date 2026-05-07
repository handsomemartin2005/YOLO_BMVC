#!/usr/bin/env bash
set -euo pipefail

python scripts/check_package.py
python visualize_dataset.py --split test --max-images 4 --tile-size 384
python train.py --epochs 1 --imgsz 640 --batch 1 --workers 0 --device auto --name smoke_idrid --exist-ok
python evaluate.py --weights runs/detect/smoke_idrid/weights/best.pt --split val --imgsz 640 --batch 1 --workers 0 --device auto --name smoke_val --exist-ok
python visualize_predictions.py --weights runs/detect/smoke_idrid/weights/best.pt --split val --imgsz 640 --max-images 4 --device auto
