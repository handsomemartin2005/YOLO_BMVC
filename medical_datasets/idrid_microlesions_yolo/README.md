# IDRiD Microlesions YOLO

Converted from the IDRiD A. Segmentation package.
Connected components in lesion masks are converted to YOLO detection boxes.
Optic-disc masks are excluded.

Source archive: https://zenodo.org/records/17219542/files/A.%20Segmentation.zip?download=1
License files from the archive are copied into metadata/.

Classes:
- 0: microaneurysm
- 1: haemorrhage
- 2: hard_exudate
- 3: soft_exudate

Splits:
- train: 43 images
- val: 11 images
- test: 27 images

Use data.yaml with Ultralytics YOLO.
