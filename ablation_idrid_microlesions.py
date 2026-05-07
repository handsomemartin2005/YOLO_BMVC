# -*- coding: utf-8 -*-
"""Run the existing A/B/C ablation controller on IDRiD microlesions."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def main() -> int:
    defaults = [
        str(ROOT / "ablation.py"),
        "--data",
        str(ROOT / "medical_datasets" / "idrid_microlesions_yolo" / "data.yaml"),
        "--project",
        str(ROOT / "runs" / "medical_ablation_idrid"),
        "--name-prefix",
        "idrid_ablation",
        "--epochs",
        "300",
        "--imgsz",
        "1280",
        "--batch",
        "4",
        "--workers",
        "4",
        "--seed",
        "3407",
        "--pretrained",
        str(ROOT / "yolov8n.pt"),
    ]
    command = [sys.executable, *defaults, *sys.argv[1:]]
    return subprocess.run(command, cwd=ROOT).returncode


if __name__ == "__main__":
    raise SystemExit(main())
