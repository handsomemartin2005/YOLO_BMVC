# -*- coding: utf-8 -*-
"""Visualize PPAP-EMA and KDE-HyperGraph feature maps from a trained checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ultralytics import YOLO
from ultralytics.nn.modules import KDEHyperGraphFusion, PPAPEMA


ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET = ROOT / "medical_datasets" / "idrid_microlesions_yolo"
DEFAULT_WEIGHTS = ROOT / "runs" / "detect" / "idrid_microlesions_full_innov" / "weights" / "best.pt"


def get_device(device: str):
    if device != "auto":
        return device
    return 0 if torch.cuda.is_available() else "cpu"


def preprocess_image(image_path: Path, imgsz: int, device):
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        raise FileNotFoundError(f"cannot read image: {image_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resize = cv2.resize(img_rgb, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
    tensor = torch.from_numpy(img_resize).permute(2, 0, 1).float() / 255.0
    return tensor.unsqueeze(0).to(device)


def tensor_map_to_numpy(x: torch.Tensor) -> np.ndarray:
    if x is None or x.ndim != 4:
        raise ValueError(f"expected 4D visualization tensor, got {None if x is None else x.shape}")
    return x[0, 0].detach().cpu().numpy()


def collect_modules(model):
    ppap_modules = []
    hg_modules = []
    for module in model.model.model.modules():
        if isinstance(module, PPAPEMA):
            module.collect_visualization = True
            ppap_modules.append(module)
        elif isinstance(module, KDEHyperGraphFusion):
            module.collect_visualization = True
            hg_modules.append(module)
    if not ppap_modules:
        raise RuntimeError("no PPAPEMA module found")
    if not hg_modules:
        raise RuntimeError("no KDEHyperGraphFusion module found")
    return ppap_modules, hg_modules


def save_figure(det_rgb, ppap_before, ppap_after, hg_before, hg_after, save_path: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes[0, 0].imshow(det_rgb)
    axes[0, 0].set_title("Detection")
    axes[0, 1].imshow(ppap_before, cmap="viridis")
    axes[0, 1].set_title("Before PPAP-EMA")
    axes[0, 2].imshow(ppap_after, cmap="viridis")
    axes[0, 2].set_title("After PPAP-EMA")
    axes[1, 0].imshow(det_rgb)
    axes[1, 0].set_title("Detection")
    axes[1, 1].imshow(hg_before, cmap="viridis")
    axes[1, 1].set_title("Before HyperGraph")
    axes[1, 2].imshow(hg_after, cmap="viridis")
    axes[1, 2].set_title("After HyperGraph")
    for ax in axes.ravel():
        ax.axis("off")
    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def parse_hg_indices(raw_index: str, hg_count: int):
    if raw_index.lower() == "all":
        return list(range(hg_count))
    index = int(raw_index)
    if index < 0 or index >= hg_count:
        raise ValueError(f"hg-index={index} out of range, found {hg_count} branches")
    return [index]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize custom model feature maps.")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--image", type=Path, default=None)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--hg-index", default="all")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--save-dir", type=Path, default=ROOT / "results" / "feature_visualization")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.weights.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.weights}. Train first or pass --weights.")
    image_path = args.image
    if image_path is None:
        images = sorted((args.dataset / "images" / args.split).glob("*.jpg"))
        if not images:
            raise RuntimeError(f"No images found for split {args.split}")
        image_path = images[args.index]
    if not image_path.exists():
        raise FileNotFoundError(image_path)

    args.save_dir.mkdir(parents=True, exist_ok=True)
    device = get_device(args.device)
    model = YOLO(str(args.weights))
    model.model.eval()
    ppap_modules, hg_modules = collect_modules(model)
    hg_indices = parse_hg_indices(args.hg_index, len(hg_modules))

    det_bgr = model.predict(source=str(image_path), imgsz=args.imgsz, conf=args.conf, device=device, save=False, verbose=False)[0].plot()
    det_rgb = cv2.cvtColor(det_bgr, cv2.COLOR_BGR2RGB)
    tensor = preprocess_image(image_path, args.imgsz, device)
    with torch.no_grad():
        _ = model.model(tensor)

    ppap_before = tensor_map_to_numpy(ppap_modules[0].latest_vis["input"])
    ppap_after = tensor_map_to_numpy(ppap_modules[0].latest_vis["output"])
    for hg_index in hg_indices:
        hg = hg_modules[hg_index]
        hg_before = tensor_map_to_numpy(hg.latest_vis["before_hg"])
        hg_after = tensor_map_to_numpy(hg.latest_vis["after_hg"])
        save_path = args.save_dir / f"{image_path.stem}_feature_vis_hg{hg_index}.png"
        save_figure(det_rgb, ppap_before, ppap_after, hg_before, hg_after, save_path)
        print(f"[OK] feature visualization saved to: {save_path}")


if __name__ == "__main__":
    main()
