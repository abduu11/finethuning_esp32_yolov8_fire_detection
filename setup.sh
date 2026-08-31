#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PY="${PYTHON:-python3}"

echo "[setup] python: $($PY --version 2>&1)"

if [ -d .venv ]; then
  echo "[setup] removing existing .venv ..."
  rm -rf .venv
fi

echo "[setup] creating venv ..."
"$PY" -m venv .venv
source .venv/bin/activate

echo "[setup] upgrading pip ..."
python -m pip install --upgrade pip setuptools wheel

echo "[setup] installing CPU-only torch + torchvision ..."
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

echo "[setup] installing ultralytics + opencv ..."
python -m pip install ultralytics opencv-python

echo "[setup] verifying ..."
python -c "import torch, torchvision, ultralytics, cv2, numpy; print('torch', torch.__version__); print('ultralytics', ultralytics.__version__); print('cv2', cv2.__version__)"

echo "[setup] OK"
