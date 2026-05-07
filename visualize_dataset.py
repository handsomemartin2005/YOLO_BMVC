# -*- coding: utf-8 -*-
"""Draw ground-truth YOLO boxes for qualitative dataset inspection."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET = ROOT / "medical_datasets" / "idrid_microlesions_yolo"
NAMES = {
    0: "microaneurysm",
    1: "haemorrhage",
    2: "hard_exudate",
    3: "soft_exudate",
}
COLORS = {
    0: (0, 220, 255),
    1: (40, 40, 230),
    2: (0, 180, 0),
    3: (220, 100, 20),
}


def read_labels(path: Path):
    labels = []
    if not path.exists():
        return labels
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        cls = int(float(parts[0]))
        x, y, w, h = [float(v) for v in parts[1:]]
        labels.append((cls, x, y, w, h))
    return labels


def draw_labels(image_path: Path, label_path: Path, tile_size: int, max_boxes: int) -> np.ndarray:
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(image_path)
    h, w = image.shape[:2]
    labels = read_labels(label_path)[:max_boxes]
    for cls, xc, yc, bw, bh in labels:
        x1 = int((xc - bw / 2) * w)
        y1 = int((yc - bh / 2) * h)
        x2 = int((xc + bw / 2) * w)
        y2 = int((yc + bh / 2) * h)
        color = COLORS.get(cls, (255, 255, 255))
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
        text = NAMES.get(cls, str(cls))
        cv2.putText(image, text, (x1, max(20, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

    scale = tile_size / max(h, w)
    resized = cv2.resize(image, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)
    canvas = np.full((tile_size, tile_size, 3), 245, dtype=np.uint8)
    y0 = (tile_size - resized.shape[0]) // 2
    x0 = (tile_size - resized.shape[1]) // 2
    canvas[y0 : y0 + resized.shape[0], x0 : x0 + resized.shape[1]] = resized
    cv2.putText(canvas, image_path.stem, (8, tile_size - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 30, 30), 1)
    return canvas


def build_montage(tiles, cols: int, gap: int) -> np.ndarray:
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
    parser = argparse.ArgumentParser(description="Visualize IDRiD YOLO ground-truth boxes.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--max-images", type=int, default=12)
    parser.add_argument("--max-boxes", type=int, default=80)
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--cols", type=int, default=3)
    parser.add_argument("--save-dir", type=Path, default=ROOT / "results" / "dataset_visualization")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_dir = args.dataset / "images" / args.split
    label_dir = args.dataset / "labels" / args.split
    image_paths = sorted(image_dir.glob("*.jpg"))[: args.max_images]
    if not image_paths:
        raise RuntimeError(f"No images found under {image_dir}")

    args.save_dir.mkdir(parents=True, exist_ok=True)
    tiles = []
    for image_path in image_paths:
        tile = draw_labels(image_path, label_dir / f"{image_path.stem}.txt", args.tile_size, args.max_boxes)
        cv2.imwrite(str(args.save_dir / f"{image_path.stem}_gt.jpg"), tile)
        tiles.append(tile)
    montage = build_montage(tiles, args.cols, gap=12)
    montage_path = args.save_dir / f"idrid_{args.split}_ground_truth_montage.jpg"
    cv2.imwrite(str(montage_path), montage)
    print(f"[OK] ground-truth montage saved to: {montage_path}")


if __name__ == "__main__":
    main()
