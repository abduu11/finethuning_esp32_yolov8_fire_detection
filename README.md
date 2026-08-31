# YOLOv8 — Détection feu/fumée pour ESP32-CAM

Fine-tuning de **YOLOv8n** pour détecter **feu** (`fire`) et **fumée** (`smoke`)
sur des images issues de caméras basse qualité (ESP32-CAM / OV2640).

## Pipeline

1. `scripts/download_dfire.py` — télécharge et prépare le dataset D-Fire
2. `scripts/augment_esp32.py` — augmente les images avec les dégradations du capteur ESP32-CAM
3. `scripts/train_finetune.py` — fine-tune YOLOv8n
4. `scripts/evaluate_model.py` — compare ancien vs nouveau modèle
5. `scripts/baseline_webcam.py` — inférence temps réel (webcam / flux)

## Installation

```bash
bash setup.sh            # ou manuellement :
python3 -m venv .venv && source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install ultralytics opencv-python
```

## Télécharger le dataset

```bash
source .venv/bin/activate
python scripts/download_dfire.py
```

## Augmentation ESP32-CAM

```bash
python scripts/augment_esp32.py          # crée data/dfire_esp32 (x3)
```

## Fine-tuning

```bash
python scripts/train_finetune.py         # GPU recommandé (~8 h sur GTX 1650)
```

Le meilleur modèle est copié dans `models/fire_smoke_esp32_yolov8n.pt`.

## Évaluation

```bash
python scripts/evaluate_model.py
```

## Inférence / Webcam

```bash
python scripts/baseline_webcam.py --model models/fire_smoke_esp32_yolov8n.pt
```

Options utiles :

| Option | Défaut | Description |
|---|---|---|
| `--conf` | `0.55` | Seuil de confiance. **Garder ≥ 0.55** pour éviter les faux positifs (flash de téléphone, lampes, soleil confondus avec le feu) |
| `--imgsz` | `416` | Taille d'inférence (`320` = plus de FPS, `640` = plus précis) |
| `--source` | `0` | Index webcam, ou chemin vidéo/image |
| `--res` | `640x480` | Résolution de capture webcam |

### Utiliser le modèle dans du code Python

```python
from ultralytics import YOLO

model = YOLO("models/fire_smoke_esp32_yolov8n.pt")
results = model.predict(source, conf=0.55, imgsz=416)
```

> ⚠️ **Important** : le seuil `conf` n'est **pas stocké** dans les poids (`.pt`).
> Il doit être passé à l'inférence. La valeur recommandée est `0.55` : plus bas
> (`0.35`) augmente fortement les faux positifs (sources lumineuses), plus haut
> (`0.65+`) réduit la détection des feux faibles.

## Résultats (jeu de test augmenté ESP32)

| Modèle | mAP50 | mAP50-95 |
|---|---|---|
| Pré-entraîné | 0.505 | 0.295 |
| **Fine-tuné ESP32** | **0.549** | **0.329** |
