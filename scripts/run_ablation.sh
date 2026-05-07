#!/usr/bin/env bash
set -euo pipefail

python ablation_idrid_microlesions.py --epochs 300 --imgsz 1280 --batch 4 --workers 4 --device 0
