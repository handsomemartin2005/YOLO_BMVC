# -*- coding: utf-8 -*-
"""Lightweight package checks for dataset labels and model construction."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ultralytics import YOLO  # noqa: E402
from ultralytics.data.utils import check_det_dataset  # noqa: E402

DATA = ROOT / "medical_datasets" / "idrid_microlesions_yolo" / "data.yaml"
MODEL = ROOT / "ultralytics" / "cfg" / "models" / "v8" / "yolov8n-full-innov.yaml"


def check_labels() -> dict:
    label_root = ROOT / "medical_datasets" / "idrid_microlesions_yolo" / "labels"
    summary = {}
    total = 0
    for split in ("train", "val", "test"):
        count = 0
        bad = []
        for path in (label_root / split).glob("*.txt"):
            for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if not line.strip():
                    continue
                parts = line.split()
                if len(parts) != 5:
                    bad.append((str(path), line_no, line))
                    continue
                cls = int(float(parts[0]))
                values = [float(v) for v in parts[1:]]
                if cls not in range(4) or any(v <= 0 or v > 1 for v in values):
                    bad.append((str(path), line_no, line))
                count += 1
        if bad:
            raise RuntimeError(f"Invalid labels in split {split}: {bad[:3]}")
        summary[split] = count
        total += count
    summary["total"] = total
    return summary


def main() -> None:
    data = check_det_dataset(str(DATA))
    labels = check_labels()
    model = YOLO(str(MODEL))
    out = {
        "data_nc": data["nc"],
        "data_names": data["names"],
        "label_counts": labels,
        "model_class": type(model).__name__,
    }
    print(json.dumps(out, indent=2))
    print("[OK] package check passed")


if __name__ == "__main__":
    main()
