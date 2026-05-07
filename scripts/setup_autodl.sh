#!/usr/bin/env bash
set -euo pipefail

python - <<'PY'
import importlib.util
missing = [name for name in ("torch", "torchvision") if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit("Missing PyTorch packages: " + ", ".join(missing) + ". Install a CUDA-compatible torch/torchvision build first.")
print("PyTorch is available.")
PY

pip install -r requirements.txt
python scripts/check_package.py
