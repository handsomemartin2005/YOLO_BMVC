# YOLO_BMVC

Minimal runnable package for BMVC-style medical small-object detection experiments with a modified YOLOv8n detector on IDRiD retinal microlesions.

## Contents

- `ultralytics/`: local YOLOv8 code with the unchanged PPAP-EMA, KDE-HyperGraph, and CLAG-RGCU modules.
- `medical_datasets/idrid_microlesions_yolo/`: converted IDRiD lesion detection dataset in YOLO format.
- `yolov8n.pt`: YOLOv8n pretrained checkpoint for partial transfer.
- `train.py`: full-model training entrypoint.
- `evaluate.py`: quantitative validation/test metrics.
- `visualize_dataset.py`: ground-truth qualitative visualization.
- `visualize_predictions.py`: prediction qualitative visualization after training.
- `visualize_innov_features.py`: PPAP-EMA and HyperGraph feature visualization after training.
- `ablation.py` and `ablation_idrid_microlesions.py`: unchanged A/B/C ablation design.

## Dataset

The included dataset is converted from IDRiD A. Segmentation. Connected components in lesion masks are converted to detection boxes. Optic-disc masks are excluded.

Classes:

```text
0 microaneurysm
1 haemorrhage
2 hard_exudate
3 soft_exudate
```

Splits:

```text
train: 43 images, 9455 boxes
val:   11 images, 1960 boxes
test:  27 images, 5774 boxes
```

Dataset config:

```text
medical_datasets/idrid_microlesions_yolo/data.yaml
```

## AutoDL Setup

On AutoDL, unzip the package and run from the repository root:

```bash
bash scripts/setup_autodl.sh
```

If your image already has a CUDA-compatible PyTorch build, this only installs the remaining dependencies. If PyTorch is missing, install the build matching your CUDA version first, for example:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

## Smoke Check

This checks dataset paths, labels, and model construction without running a long training job:

```bash
python scripts/check_package.py
```

Quick one-epoch run:

```bash
python train.py --epochs 1 --imgsz 640 --batch 1 --workers 0 --device 0 --amp false --name smoke_idrid --exist-ok
```

## Full Training

The model architecture is unchanged from the previous project.

```bash
python train.py --epochs 300 --imgsz 1280 --batch 4 --workers 4 --device 0 --amp false
```

Outputs are written to:

```text
runs/detect/idrid_microlesions_full_innov
```

## Quantitative Results

Evaluate the trained best checkpoint on the test split:

```bash
python evaluate.py --split test
```

Metrics are saved under:

```text
runs/val/idrid_microlesions_test
```

## Qualitative Visualization

Ground-truth preview:

```bash
python visualize_dataset.py --split test --max-images 12
```

Prediction montage after training:

```bash
python visualize_predictions.py --split test --conf 0.25
```

Feature visualization after training:

```bash
python visualize_innov_features.py --split test --index 0 --hg-index all
```

## Ablation

Generate all A/B/C ablation YAMLs without training:

```bash
python ablation_idrid_microlesions.py --dry-run
```

Run all ablations:

```bash
python ablation_idrid_microlesions.py --epochs 300 --imgsz 1280 --batch 4 --workers 4 --device 0
```

Run a quick ablation smoke test:

```bash
python ablation_idrid_microlesions.py --variants base ABC --epochs 1 --imgsz 640 --batch 1 --workers 0 --device auto --exist-ok
```

## Notes

- The package intentionally excludes raw IDRiD archives and old industrial-defect runs.
- The converted IDRiD dataset license files are in `medical_datasets/idrid_microlesions_yolo/metadata/`.
- Use commands from the repository root so Python imports the bundled local `ultralytics` package.
