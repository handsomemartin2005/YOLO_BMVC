# -*- coding: utf-8 -*-
"""Train the unchanged full innovation YOLOv8n model on IDRiD microlesions."""

from __future__ import annotations

import argparse
from multiprocessing import freeze_support
from pathlib import Path

import torch
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT / "ultralytics" / "cfg" / "models" / "v8" / "yolov8n-full-innov.yaml"
DEFAULT_DATA = ROOT / "medical_datasets" / "idrid_microlesions_yolo" / "data.yaml"
DEFAULT_PROJECT = ROOT / "runs" / "detect"
DEFAULT_PRETRAINED = ROOT / "yolov8n.pt"


def parse_pretrained(value: str):
    lower = str(value).strip().lower()
    if lower in {"true", "1", "yes", "y"}:
        return True
    if lower in {"false", "0", "no", "n"}:
        return False
    if lower in {"none", "null"}:
        return None
    return value


def str2bool(value):
    if isinstance(value, bool):
        return value
    lower = str(value).strip().lower()
    if lower in {"true", "1", "yes", "y", "on"}:
        return True
    if lower in {"false", "0", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"expected a boolean value, got '{value}'")


def get_device(device: str):
    if device != "auto":
        return device
    return 0 if torch.cuda.is_available() else "cpu"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train full innovation YOLOv8n on IDRiD microlesion detection.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--project", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--name", default="idrid_microlesions_full_innov")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--pretrained", default=str(DEFAULT_PRETRAINED))
    parser.add_argument("--save-period", type=int, default=10)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--cache", action="store_true")
    parser.add_argument("--exist-ok", action="store_true")
    parser.add_argument("--amp", type=str2bool, default=True)
    parser.add_argument("--deterministic", type=str2bool, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.data.exists():
        raise FileNotFoundError(f"Dataset config not found: {args.data}")
    if not args.model.exists():
        raise FileNotFoundError(f"Model YAML not found: {args.model}")

    print("torch version:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    print("cuda count:", torch.cuda.device_count())
    print("model:", args.model)
    print("data:", args.data)

    model = YOLO(str(args.model))
    model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=get_device(args.device),
        project=str(args.project),
        name=args.name,
        exist_ok=args.exist_ok,
        pretrained=parse_pretrained(args.pretrained),
        save=True,
        save_period=args.save_period,
        cache=args.cache,
        patience=args.patience if args.patience is not None else args.epochs,
        seed=args.seed,
        deterministic=args.deterministic,
        verbose=True,
        resume=False,
        amp=args.amp,
    )


if __name__ == "__main__":
    freeze_support()
    main()
