# -*- coding: utf-8 -*-
"""Create qualitative prediction images and a montage for trained checkpoints."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET = ROOT / "medical_datasets" / "idrid_microlesions_yolo"
DEFAULT_WEIGHTS = ROOT / "runs" / "detect" / "idrid_microlesions_full_innov" / "weights" / "best.pt"


def get_device(device: str):
    if device != "auto":
        return device
    return 0 if torch.cuda.is_available() else "cpu"


def resize_tile(image: np.ndarray, tile_size: int) -> np.ndarray:
    h, w = image.shape[:2]
    scale = tile_size / max(h, w)
    resized = cv2.resize(image, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)
    canvas = np.full((tile_size, tile_size, 3), 245, dtype=np.uint8)
    y0 = (tile_size - resized.shape[0]) // 2
    x0 = (tile_size - resized.shape[1]) // 2
    canvas[y0 : y0 + resized.shape[0], x0 : x0 + resized.shape[1]] = resized
    return canvas


def montage(tiles, cols: int, gap: int) -> np.ndarray:
    rows = math.ceil(len(tiles) / cols)
    size = tiles[0].shape[0]
    canvas = np.full((rows * size + (rows - 1) * gap, cols * size + (cols - 1) * gap, 3), 255, dtype=np.uint8)
    for i, tile in enumerate(tiles):
        r, c = divmod(i, cols)
        y = r * (size + gap)
        x = c * (size + gap)
        canvas[y : y + size, x : x + size] = tile
    return canvas


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize IDRiD predictions.")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--max-images", type=int, default=12)
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--cols", type=int, default=3)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--save-dir", type=Path, default=ROOT / "results" / "prediction_visualization")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.weights.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.weights}. Train first or pass --weights.")
    image_paths = sorted((args.dataset / "images" / args.split).glob("*.jpg"))[: args.max_images]
    if not image_paths:
        raise RuntimeError(f"No images found for split {args.split}")

    args.save_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(args.weights))
    results = model.predict(
        source=[str(p) for p in image_paths],
        imgsz=args.imgsz,
        conf=args.conf,
        device=get_device(args.device),
        save=False,
        verbose=False,
    )

    tiles = []
    for result in results:
        plotted = result.plot()
        tile = resize_tile(plotted, args.tile_size)
        out_path = args.save_dir / f"{Path(result.path).stem}_pred.jpg"
        cv2.imwrite(str(out_path), tile)
        tiles.append(tile)
    montage_path = args.save_dir / f"idrid_{args.split}_prediction_montage.jpg"
    cv2.imwrite(str(montage_path), montage(tiles, args.cols, gap=12))
    print(f"[OK] prediction montage saved to: {montage_path}")


if __name__ == "__main__":
    main()
