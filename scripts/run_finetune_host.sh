#!/usr/bin/env bash
# Fine-tuning YOLOv8n (ESP32 fire/smoke) — à exécuter sur la MACHINE HÔTE
# (hors sandbox Flatpak, pour avoir accès au GPU GTX 1650).
#
# Usage:  bash scripts/run_finetune_host.sh [--skip-setup]
#   --skip-setup : réutilise l'environnement déjà installé (venv + deps).
#
set -euo pipefail

cd "$(dirname "$0")/.."

PY="${PYTHON:-python3}"

echo "=============================================="
echo " Fine-tuning YOLOv8n — fire/smoke ESP32-CAM"
echo "=============================================="

# ---- 0. Vérifier le GPU ---------------------------------------------------
echo "[1/4] Vérification GPU ..."
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "  WARN: nvidia-smi introuvable. On vérifie quand même via torch plus bas."
fi

# ---- 1. Environnement -----------------------------------------------------
if [[ "${1:-}" == "--skip-setup" ]]; then
  echo "[2/4] Setup ignoré (--skip-setup)."
else
  echo "[2/4] Préparation de l'environnement ..."

  # Recréer le venv seulement s'il ne correspond pas au python système
  # (le venv actuel a été créé depuis le sandbox Flatpak → inutilisable sur host).
  if [[ -d .venv ]]; then
    echo "  Suppression de l'ancien .venv (créé dans le sandbox) ..."
    rm -rf .venv
  fi

  echo "  Création du venv avec: $($PY --version 2>&1) ..."
  "$PY" -m venv .venv
  source .venv/bin/activate

  echo "  Installation pip/torch/torchvision (CUDA) ..."
  python -m pip install --upgrade pip setuptools wheel
  python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

  echo "  Installation ultralytics + opencv ..."
  python -m pip install ultralytics opencv-python
fi

source .venv/bin/activate

# ---- 2. Vérifier CUDA depuis torch ---------------------------------------
echo "[3/4] Vérification CUDA via torch ..."
python - <<'PY'
import torch
print("  torch:", torch.__version__)
print("  CUDA disponible:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("  GPU:", torch.cuda.get_device_name(0))
    print("  VRAM (Go):", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1))
else:
    print("  ERREUR: CUDA indisponible — le training GPU ne pourra pas se lancer.")
    raise SystemExit(1)
PY

# ---- 3. Lancement du fine-tuning ------------------------------------------
echo "[4/4] Lancement du fine-tuning ..."
echo "  Dataset : data/dfire_esp32 (42k images train, augmentées ESP32)"
echo "  Modèle  : yolov8n.pt (COCO pretrained)"
echo ""
python scripts/train_finetune.py

echo ""
echo "=============================================="
echo " Fine-tuning terminé."
echo " Évaluation comparative: python scripts/evaluate_model.py"
echo "=============================================="
