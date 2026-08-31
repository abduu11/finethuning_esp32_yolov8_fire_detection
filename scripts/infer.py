#!/usr/bin/env python3
"""Wrapper d'inférence pour le modèle feu/fumée ESP32-CAM.

Force un seuil de confiance par défaut à 0.55 pour éviter les faux positifs
(flash de téléphone, lampes, soleil). Utilisation :

    python infer.py --source image.jpg
    python infer.py --source video.mp4
    python infer.py --source dossier/
    python infer.py --source 0            # webcam
    python infer.py --source 0 --save     # enregistrer les résultats annotés
"""

import argparse
import torch
from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser(description="Fire/smoke inference (ESP32-CAM model)")
    p.add_argument("--source", required=True,
                   help="image/video path, folder, or webcam index (0)")
    p.add_argument("--model", default="models/fire_smoke_esp32_yolov8n.pt",
                   help="model weights")
    p.add_argument("--conf", type=float, default=0.55,
                   help="confidence threshold (default 0.55 to limit false positives)")
    p.add_argument("--imgsz", type=int, default=416, help="inference size")
    p.add_argument("--device", default="auto", help="auto / 0 / cpu")
    p.add_argument("--save", action="store_true", help="save annotated results")
    return p.parse_args()


def main():
    args = parse_args()
    if args.device == "auto":
        args.device = "0" if torch.cuda.is_available() else "cpu"

    model = YOLO(args.model)
    source = int(args.source) if args.source.isdigit() else args.source

    results = model.predict(
        source=source,
        conf=args.conf,
        imgsz=args.imgsz,
        device=args.device,
        save=args.save,
        show=args.source.isdigit(),  # afficher la fenêtre pour la webcam
    )

    print(f"\n[infer] conf={args.conf} imgsz={args.imgsz} device={args.device}")
    for r in results:
        if r.boxes is None or len(r.boxes) == 0:
            print(f"[infer] {r.path}: no detection")
            continue
        for b in r.boxes:
            name = model.names[int(b.cls[0])]
            print(f"[infer] {name}: {float(b.conf[0]):.2f}")


if __name__ == "__main__":
    main()
