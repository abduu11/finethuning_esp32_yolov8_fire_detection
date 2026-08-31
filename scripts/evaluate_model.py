#!/usr/bin/env python3
"""Compare old vs new fine-tuned model on the ESP32-augmented test set.

Runs validation on both models and prints a side-by-side comparison
of mAP50, mAP50-95, precision, and recall.
"""

import argparse

from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser(description="Compare old vs new model on test data")
    p.add_argument("--old", default="models/fire_smoke_yolov8n.pt", help="old model")
    p.add_argument("--new", default="models/fire_smoke_esp32_yolov8n.pt", help="new fine-tuned model")
    p.add_argument("--data", default="data/dfire_esp32/data.yaml", help="dataset config")
    p.add_argument("--imgsz", type=int, default=416)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--device", default="0")
    p.add_argument("--conf", type=float, default=0.35)
    return p.parse_args()


def evaluate(model_path, data, imgsz, batch, device, conf, label):
    """Run validation and return metrics dict."""
    print(f"\n{'=' * 60}")
    print(f"Evaluating: {label}")
    print(f"Model: {model_path}")
    print(f"{'=' * 60}")

    model = YOLO(model_path)
    metrics = model.val(
        data=data,
        imgsz=imgsz,
        batch=batch,
        device=device,
        conf=conf,
        verbose=True,
        plots=True,
    )

    return {
        "mAP50": metrics.box.map50,
        "mAP50-95": metrics.box.map,
        "precision": metrics.box.mp,
        "recall": metrics.box.mr,
    }


def main():
    args = parse_args()

    print("=" * 60)
    print("MODEL COMPARISON: Old vs New (ESP32 Fine-Tuned)")
    print("=" * 60)

    old_metrics = evaluate(
        args.old, args.data, args.imgsz, args.batch,
        args.device, args.conf, "OLD MODEL (pre-trained)"
    )
    new_metrics = evaluate(
        args.new, args.data, args.imgsz, args.batch,
        args.device, args.conf, "NEW MODEL (ESP32 fine-tuned)"
    )

    # Print comparison table
    print(f"\n{'=' * 60}")
    print("COMPARISON SUMMARY")
    print(f"{'=' * 60}")
    print(f"{'Metric':<15} {'Old':>10} {'New':>10} {'Delta':>10}")
    print("-" * 50)

    for key in old_metrics:
        old_val = old_metrics[key]
        new_val = new_metrics[key]
        delta = new_val - old_val
        sign = "+" if delta >= 0 else ""
        print(f"{key:<15} {old_val:>10.4f} {new_val:>10.4f} {sign}{delta:>9.4f}")

    print(f"\n{'=' * 60}")
    if new_metrics["mAP50"] > old_metrics["mAP50"]:
        print("✅ NEW model is BETTER on the ESP32-augmented test set!")
    else:
        print("⚠️  OLD model still performs better. Consider:")
        print("    - More training epochs")
        print("    - More augmented copies (--copies 3 or 4)")
        print("    - Adjusting learning rate")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
