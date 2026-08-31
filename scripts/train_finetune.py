#!/usr/bin/env python3
"""Fine-tune YOLOv8n on ESP32-CAM augmented fire/smoke dataset.

Optimized for GTX 1650 (4 GB VRAM):
  - batch=8, imgsz=416
  - AdamW optimizer with warmup
  - Early stopping with patience=10
"""

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune YOLOv8n for ESP32-CAM fire/smoke detection")
    p.add_argument("--model", default="yolov8n.pt", help="base model weights (COCO pretrained)")
    p.add_argument("--data", default="data/dfire_esp32/data.yaml", help="dataset config")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--imgsz", type=int, default=416)
    p.add_argument("--batch", type=int, default=8, help="batch size (8 for GTX 1650 4GB)")
    p.add_argument("--device", default="0", help="CUDA device")
    p.add_argument("--patience", type=int, default=10, help="early stopping patience")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--project", default="runs/finetune")
    p.add_argument("--name", default="esp32_fire_smoke")
    p.add_argument("--resume", action="store_true", help="resume from last checkpoint")
    return p.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("YOLOv8n Fine-Tuning for ESP32-CAM Fire/Smoke Detection")
    print("=" * 60)
    print(f"  Base model:  {args.model}")
    print(f"  Dataset:     {args.data}")
    print(f"  Epochs:      {args.epochs}")
    print(f"  Image size:  {args.imgsz}")
    print(f"  Batch size:  {args.batch}")
    print(f"  Device:      {args.device}")
    print(f"  Patience:    {args.patience}")
    print(f"  Output:      {args.project}/{args.name}")
    print("=" * 60)

    # Verify dataset exists
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"\nERROR: Dataset config not found: {args.data}")
        print("Run these scripts first:")
        print("  python scripts/download_dfire.py")
        print("  python scripts/augment_esp32.py")
        return

    # Load model
    model = YOLO(args.model)

    # Resolve project to an absolute path so ultralytics does not
    # prepend its default "runs/detect" prefix to relative paths.
    project = Path(args.project).resolve()

    # Train
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        patience=args.patience,
        workers=args.workers,
        project=str(project),
        name=args.name,
        resume=args.resume,
        # Augmentation params (on top of our ESP32 augmentation)
        hsv_h=0.02,
        hsv_s=0.7,
        hsv_v=0.5,
        flipud=0.3,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
        # Optimizer
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        weight_decay=0.0005,
        warmup_epochs=3,
        # Other
        exist_ok=True,
        verbose=True,
        plots=True,
        save=True,
        save_period=10,
    )

    # Copy best model to models/
    best_pt = project / args.name / "weights" / "best.pt"
    dst_pt = Path("models/fire_smoke_esp32_yolov8n.pt")
    if best_pt.exists():
        shutil.copy2(best_pt, dst_pt)
        print(f"\n[train] Best model copied to: {dst_pt}")

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Best weights:  {best_pt}")
    print(f"Deployed copy: {dst_pt}")
    print(f"Training logs: {project}/{args.name}/")
    print()
    print("Next steps:")
    print("  1. Evaluate: python scripts/evaluate_model.py")
    print("  2. Test:     python scripts/baseline_webcam.py --model models/fire_smoke_esp32_yolov8n.pt")


if __name__ == "__main__":
    main()
