# -*- coding: utf-8 -*-
"""Prepare IDRiD lesion masks as a YOLO detection dataset.

The IDRiD segmentation package contains pixel masks for diabetic-retinopathy
lesions. This script converts each connected mask component into one YOLO bbox.
The optic-disc masks are intentionally ignored because they are anatomical
context, not small lesion targets.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = ROOT / "medical_datasets"
DEFAULT_DATASET_NAME = "idrid_microlesions_yolo"
DEFAULT_URL = "https://zenodo.org/records/17219542/files/A.%20Segmentation.zip?download=1"


@dataclass(frozen=True)
class LesionClass:
    class_id: int
    name: str
    folder: str
    suffix: str


LESION_CLASSES: Tuple[LesionClass, ...] = (
    LesionClass(0, "microaneurysm", "1. Microaneurysms", "MA"),
    LesionClass(1, "haemorrhage", "2. Haemorrhages", "HE"),
    LesionClass(2, "hard_exudate", "3. Hard Exudates", "EX"),
    LesionClass(3, "soft_exudate", "4. Soft Exudates", "SE"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert IDRiD segmentation masks to YOLO boxes.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--archive", type=Path, default=None)
    parser.add_argument("--raw-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--min-area", type=int, default=1, help="Minimum connected-component area in mask pixels.")
    parser.add_argument("--overwrite", action="store_true", help="Rebuild output dataset if it already exists.")
    parser.add_argument("--no-download", action="store_true", help="Fail instead of downloading if the archive is missing.")
    return parser.parse_args()


def progress(block_count: int, block_size: int, total_size: int) -> None:
    if total_size <= 0:
        return
    downloaded = min(block_count * block_size, total_size)
    pct = downloaded * 100.0 / total_size
    sys.stdout.write(f"\rDownloading: {pct:5.1f}%")
    sys.stdout.flush()
    if downloaded >= total_size:
        sys.stdout.write("\n")


def download_archive(url: str, archive_path: Path, no_download: bool) -> None:
    if archive_path.exists() and archive_path.stat().st_size > 0:
        print(f"Archive exists: {archive_path}")
        return
    if no_download:
        raise FileNotFoundError(f"Archive missing and --no-download was set: {archive_path}")
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading IDRiD archive to {archive_path}")
    urllib.request.urlretrieve(url, archive_path, reporthook=progress)


def extract_archive(archive_path: Path, raw_dir: Path) -> None:
    expected = raw_dir / "A. Segmentation"
    if expected.exists():
        print(f"Raw directory exists: {expected}")
        return
    raw_dir.mkdir(parents=True, exist_ok=True)
    print(f"Extracting {archive_path} to {raw_dir}")
    with zipfile.ZipFile(archive_path) as zf:
        zf.extractall(raw_dir)


def split_ids(training_ids: Sequence[str], val_ratio: float, seed: int) -> Tuple[List[str], List[str]]:
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("--val-ratio must be between 0 and 1")
    rng = np.random.default_rng(seed)
    ids = np.array(sorted(training_ids))
    order = rng.permutation(len(ids))
    val_count = max(1, int(round(len(ids) * val_ratio)))
    val_idx = set(order[:val_count].tolist())
    train, val = [], []
    for idx, image_id in enumerate(ids.tolist()):
        (val if idx in val_idx else train).append(image_id)
    return train, val


def list_image_ids(image_dir: Path) -> List[str]:
    return sorted(p.stem for p in image_dir.glob("IDRiD_*.jpg"))


def mask_to_boxes(mask_path: Path, width: int, height: int, min_area: int) -> List[Tuple[float, float, float, float]]:
    if not mask_path.exists():
        return []
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f"Could not read mask: {mask_path}")
    mask = (mask > 0).astype(np.uint8)
    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

    boxes: List[Tuple[float, float, float, float]] = []
    for label in range(1, count):
        x, y, w, h, area = stats[label]
        if area < min_area or w <= 0 or h <= 0:
            continue
        x_center = (x + w / 2.0) / width
        y_center = (y + h / 2.0) / height
        boxes.append((x_center, y_center, w / width, h / height))
    return boxes


def ensure_clean_output(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Output dataset already exists, pass --overwrite to rebuild: {output_dir}")
        if overwrite:
            shutil.rmtree(output_dir)
    for split in ("train", "val", "test"):
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
    (output_dir / "metadata").mkdir(parents=True, exist_ok=True)


def copy_license_files(source_root: Path, output_dir: Path) -> None:
    for name in ("LICENSE.txt", "CC-BY-4.0.txt"):
        source = source_root / name
        if source.exists():
            shutil.copy2(source, output_dir / "metadata" / name)


def format_label(class_id: int, box: Tuple[float, float, float, float]) -> str:
    return f"{class_id} " + " ".join(f"{value:.8f}" for value in box)


def convert_split(
    image_ids: Iterable[str],
    split_name: str,
    source_split_name: str,
    source_root: Path,
    output_dir: Path,
    min_area: int,
) -> Dict[str, int]:
    image_dir = source_root / "1. Original Images" / source_split_name
    mask_root = source_root / "2. All Segmentation Groundtruths" / source_split_name
    counts = {cls.name: 0 for cls in LESION_CLASSES}
    image_count = 0
    empty_images = 0

    for image_id in sorted(image_ids):
        src_image = image_dir / f"{image_id}.jpg"
        if not src_image.exists():
            raise FileNotFoundError(src_image)
        dst_image = output_dir / "images" / split_name / src_image.name
        shutil.copy2(src_image, dst_image)

        with Image.open(src_image) as image:
            width, height = image.size

        labels: List[str] = []
        for cls in LESION_CLASSES:
            mask_path = mask_root / cls.folder / f"{image_id}_{cls.suffix}.tif"
            boxes = mask_to_boxes(mask_path, width, height, min_area)
            counts[cls.name] += len(boxes)
            labels.extend(format_label(cls.class_id, box) for box in boxes)

        label_path = output_dir / "labels" / split_name / f"{image_id}.txt"
        label_path.write_text("\n".join(labels) + ("\n" if labels else ""), encoding="utf-8")
        image_count += 1
        if not labels:
            empty_images += 1

    counts["images"] = image_count
    counts["empty_images"] = empty_images
    counts["objects"] = sum(counts[cls.name] for cls in LESION_CLASSES)
    return counts


def write_data_yaml(output_dir: Path) -> None:
    yaml_path = output_dir / "data.yaml"
    names = "\n".join(f"  {cls.class_id}: {cls.name}" for cls in LESION_CLASSES)
    text = (
        f"path: {output_dir.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "nc: 4\n"
        "names:\n"
        f"{names}\n"
    )
    yaml_path.write_text(text, encoding="utf-8")


def write_readme(output_dir: Path, summary: dict, url: str) -> None:
    lines = [
        "# IDRiD Microlesions YOLO",
        "",
        "Converted from the IDRiD A. Segmentation package.",
        "Connected components in lesion masks are converted to YOLO detection boxes.",
        "Optic-disc masks are excluded.",
        "",
        f"Source archive: {url}",
        "License files from the archive are copied into metadata/.",
        "",
        "Classes:",
    ]
    lines.extend(f"- {cls.class_id}: {cls.name}" for cls in LESION_CLASSES)
    lines.extend(
        [
            "",
            "Splits:",
            f"- train: {summary['splits']['train']['images']} images",
            f"- val: {summary['splits']['val']['images']} images",
            f"- test: {summary['splits']['test']['images']} images",
            "",
            "Use data.yaml with Ultralytics YOLO.",
        ]
    )
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    archive = (args.archive or root / "archives" / "IDRiD_A_Segmentation.zip").resolve()
    raw_dir = (args.raw_dir or root / "idrid_raw").resolve()
    output_dir = (args.output_dir or root / args.dataset_name).resolve()

    download_archive(args.url, archive, args.no_download)
    extract_archive(archive, raw_dir)

    source_root = raw_dir / "A. Segmentation"
    train_image_dir = source_root / "1. Original Images" / "a. Training Set"
    test_image_dir = source_root / "1. Original Images" / "b. Testing Set"
    training_ids = list_image_ids(train_image_dir)
    testing_ids = list_image_ids(test_image_dir)
    train_ids, val_ids = split_ids(training_ids, args.val_ratio, args.seed)

    ensure_clean_output(output_dir, args.overwrite)
    for split in ("train", "val", "test"):
        for child in (output_dir / "images" / split).glob("*"):
            child.unlink()
        for child in (output_dir / "labels" / split).glob("*"):
            child.unlink()

    split_map = {
        "train": (train_ids, "a. Training Set"),
        "val": (val_ids, "a. Training Set"),
        "test": (testing_ids, "b. Testing Set"),
    }
    summary = {
        "dataset": "IDRiD A. Segmentation",
        "url": args.url,
        "output_dir": str(output_dir),
        "val_ratio": args.val_ratio,
        "seed": args.seed,
        "min_area": args.min_area,
        "classes": {cls.class_id: cls.name for cls in LESION_CLASSES},
        "splits": {},
    }
    for split, (ids, source_split) in split_map.items():
        summary["splits"][split] = convert_split(ids, split, source_split, source_root, output_dir, args.min_area)

    write_data_yaml(output_dir)
    copy_license_files(source_root, output_dir)
    (output_dir / "metadata" / "split.json").write_text(json.dumps(split_map, indent=2), encoding="utf-8")
    (output_dir / "metadata" / "conversion_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_readme(output_dir, summary, args.url)

    print(json.dumps(summary["splits"], indent=2))
    print(f"YOLO data config: {output_dir / 'data.yaml'}")


if __name__ == "__main__":
    main()
