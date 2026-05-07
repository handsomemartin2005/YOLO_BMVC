# -*- coding: utf-8 -*-
"""Quantitative validation/test evaluation for IDRiD microlesion checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT / "medical_datasets" / "idrid_microlesions_yolo" / "data.yaml"
DEFAULT_WEIGHTS = ROOT / "runs" / "detect" / "idrid_microlesions_full_innov" / "weights" / "best.pt"


def get_device(device: str):
    if device != "auto":
        return device
    return 0 if torch.cuda.is_available() else "cpu"


def to_float(value):
    try:
        return float(value)
    except Exception:
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained IDRiD microlesion detector.")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--project", type=Path, default=ROOT / "runs" / "val")
    parser.add_argument("--name", default=None)
    parser.add_argument("--exist-ok", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.weights.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.weights}. Train first or pass --weights.")
    if not args.data.exists():
        raise FileNotFoundError(f"Dataset config not found: {args.data}")

    name = args.name or f"idrid_microlesions_{args.split}"
    model = YOLO(str(args.weights))
    metrics = model.val(
        data=str(args.data),
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=get_device(args.device),
        project=str(args.project),
        name=name,
        exist_ok=args.exist_ok,
        plots=True,
        save_json=False,
        verbose=True,
    )

    save_dir = Path(getattr(metrics, "save_dir", args.project / name))
    summary = {
        "weights": str(args.weights),
        "data": str(args.data),
        "split": args.split,
        "imgsz": args.imgsz,
        "map50_95": to_float(getattr(metrics.box, "map", None)),
        "map50": to_float(getattr(metrics.box, "map50", None)),
        "map75": to_float(getattr(metrics.box, "map75", None)),
        "per_class_map50_95": [to_float(v) for v in getattr(metrics.box, "maps", [])],
        "results_dict": {k: to_float(v) for k, v in getattr(metrics, "results_dict", {}).items()},
    }
    save_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_dir / "metrics_summary.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[OK] metrics summary saved to: {out_path}")


if __name__ == "__main__":
    main()
