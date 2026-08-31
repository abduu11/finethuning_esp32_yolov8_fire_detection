#!/usr/bin/env python3
"""Augment a YOLO dataset with ESP32-CAM (OV2640) sensor degradation.

Applies 6 types of degradation to simulate the low-quality images
produced by the ESP32-CAM module:
  1. Heavy JPEG compression (quality 10-40)
  2. Gaussian noise (sigma 15-40)
  3. Color shift / yellow-green tint
  4. Low resolution (downscale to 320x240 then upscale)
  5. Gaussian blur (kernel 3-7)
  6. Exposure variation (gamma 0.5-1.8)

Each degradation is applied independently with 50% probability,
creating realistic variation in the output images.
"""

import argparse
import random
import shutil
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# ESP32-CAM degradation functions
# ---------------------------------------------------------------------------

def jpeg_compress(img: np.ndarray, rng: random.Random) -> np.ndarray:
    """Simulate heavy JPEG compression (quality 10-40)."""
    quality = rng.randint(10, 40)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, buf = cv2.imencode(".jpg", img, encode_param)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def gaussian_noise(img: np.ndarray, rng: random.Random) -> np.ndarray:
    """Add Gaussian noise (sigma 15-40)."""
    sigma = rng.uniform(15, 40)
    noise = np.random.default_rng(rng.randint(0, 2**31)).normal(0, sigma, img.shape).astype(np.float32)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def color_shift(img: np.ndarray, rng: random.Random) -> np.ndarray:
    """Simulate OV2640 yellow/green color cast."""
    img = img.astype(np.float32)
    img[:, :, 1] += rng.uniform(10, 30)   # green channel (BGR)
    img[:, :, 2] += rng.uniform(5, 20)    # red channel (BGR)
    return np.clip(img, 0, 255).astype(np.uint8)


def low_resolution(img: np.ndarray, rng: random.Random) -> np.ndarray:
    """Downscale to 320x240 (QVGA) and upscale back."""
    h, w = img.shape[:2]
    # Random low res: 240p to 480p
    target_h = rng.choice([240, 320, 480])
    target_w = int(target_h * w / h)
    small = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def blur(img: np.ndarray, rng: random.Random) -> np.ndarray:
    """Apply Gaussian blur (kernel 3-7)."""
    k = rng.choice([3, 5, 7])
    return cv2.GaussianBlur(img, (k, k), 0)


def exposure_variation(img: np.ndarray, rng: random.Random) -> np.ndarray:
    """Apply gamma correction (0.5 = dark, 1.8 = bright)."""
    gamma = rng.uniform(0.5, 1.8)
    inv_gamma = 1.0 / gamma
    lut = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype(np.uint8)
    return cv2.LUT(img, lut)


# All degradation functions with 50% probability each
DEGRADATIONS = [
    ("jpeg_compress", jpeg_compress, 0.5),
    ("gaussian_noise", gaussian_noise, 0.5),
    ("color_shift", color_shift, 0.5),
    ("low_resolution", low_resolution, 0.5),
    ("blur", blur, 0.5),
    ("exposure_variation", exposure_variation, 0.5),
]


def apply_esp32_degradation(img: np.ndarray, seed: int) -> np.ndarray:
    """Apply random ESP32-CAM degradations to an image."""
    rng = random.Random(seed)
    for name, func, prob in DEGRADATIONS:
        if rng.random() < prob:
            img = func(img, rng)
    return img


# ---------------------------------------------------------------------------
# Worker function for multiprocessing
# ---------------------------------------------------------------------------

def process_image(task: tuple) -> dict:
    """Process a single image: create augmented copies + keep original."""
    img_path, lbl_path, dst_img_dir, dst_lbl_dir, n_copies, base_seed = task

    results = {"original": 0, "augmented": 0, "errors": 0}

    try:
        img = cv2.imread(str(img_path))
        if img is None:
            results["errors"] = 1
            return results

        stem = img_path.stem
        suffix = img_path.suffix

        # 1. Copy original image (keep webcam capability)
        dst_orig_img = dst_img_dir / f"{stem}_orig{suffix}"
        cv2.imwrite(str(dst_orig_img), img)

        # Copy original label
        if lbl_path and lbl_path.exists():
            dst_orig_lbl = dst_lbl_dir / f"{stem}_orig.txt"
            shutil.copy2(lbl_path, dst_orig_lbl)
        results["original"] = 1

        # 2. Create augmented copies
        for i in range(n_copies):
            seed = base_seed + hash(f"{stem}_{i}") % (2**31)
            aug_img = apply_esp32_degradation(img.copy(), seed)

            dst_aug_img = dst_img_dir / f"{stem}_esp32_{i}.jpg"
            cv2.imwrite(str(dst_aug_img), aug_img, [cv2.IMWRITE_JPEG_QUALITY, 85])

            if lbl_path and lbl_path.exists():
                dst_aug_lbl = dst_lbl_dir / f"{stem}_esp32_{i}.txt"
                shutil.copy2(lbl_path, dst_aug_lbl)

            results["augmented"] += 1

    except Exception as e:
        print(f"  [error] {img_path.name}: {e}")
        results["errors"] = 1

    return results


def augment_split(split: str, src_dir: Path, dst_dir: Path, n_copies: int, seed: int, workers: int):
    """Augment all images in a dataset split."""
    src_img_dir = src_dir / split / "images"
    src_lbl_dir = src_dir / split / "labels"
    dst_img_dir = dst_dir / split / "images"
    dst_lbl_dir = dst_dir / split / "labels"

    if not src_img_dir.exists():
        print(f"  [skip] {split}: no images found at {src_img_dir}")
        return

    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_lbl_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(list(src_img_dir.glob("*.jpg")) + list(src_img_dir.glob("*.png")))
    print(f"  [{split}] Processing {len(images)} images (x{n_copies} augmented + originals) ...")

    # Prepare tasks
    tasks = []
    for img_path in images:
        lbl_path = src_lbl_dir / (img_path.stem + ".txt")
        tasks.append((img_path, lbl_path, dst_img_dir, dst_lbl_dir, n_copies, seed))

    # Process with multiprocessing
    total_orig = 0
    total_aug = 0
    total_err = 0

    with Pool(processes=workers) as pool:
        for i, result in enumerate(pool.imap_unordered(process_image, tasks, chunksize=50)):
            total_orig += result["original"]
            total_aug += result["augmented"]
            total_err += result["errors"]

            if (i + 1) % 500 == 0:
                print(f"    ... {i+1}/{len(tasks)} done ({total_aug} augmented)")

    print(f"  [{split}] Done: {total_orig} originals + {total_aug} augmented ({total_err} errors)")


def create_data_yaml(dst_dir: Path):
    """Create data.yaml for the augmented dataset."""
    yaml_content = f"""# D-Fire + ESP32-CAM augmentation dataset
# Original images + synthetic ESP32 degradation
# Classes: 0 = smoke, 1 = fire

path: {dst_dir.resolve()}
train: train/images
val: val/images
test: test/images

names:
  0: smoke
  1: fire
"""
    yaml_path = dst_dir / "data.yaml"
    yaml_path.write_text(yaml_content)
    print(f"[augment] Created {yaml_path}")


def parse_args():
    p = argparse.ArgumentParser(description="Augment dataset with ESP32-CAM degradation")
    p.add_argument("--src", default="data/dfire", help="source dataset directory")
    p.add_argument("--dst", default="data/dfire_esp32", help="destination directory")
    p.add_argument("--copies", type=int, default=2, help="augmented copies per image (default: 2)")
    p.add_argument("--seed", type=int, default=42, help="random seed")
    p.add_argument("--workers", type=int, default=4, help="parallel workers")
    return p.parse_args()


def main():
    args = parse_args()
    src_dir = Path(args.src)
    dst_dir = Path(args.dst)

    if not src_dir.exists():
        print(f"[augment] ERROR: Source directory not found: {src_dir}")
        print("[augment] Run download_dfire.py first!")
        return

    if dst_dir.exists():
        print(f"[augment] Output directory already exists: {dst_dir}")
        print("[augment] To re-augment, delete it first.")
        return

    print(f"[augment] Source: {src_dir}")
    print(f"[augment] Destination: {dst_dir}")
    print(f"[augment] Copies per image: {args.copies}")
    print(f"[augment] Workers: {args.workers}")
    print(f"[augment] Each image -> 1 original + {args.copies} ESP32-degraded = {1 + args.copies}x")
    print()

    # Augment each split
    for split in ["train", "val", "test"]:
        augment_split(split, src_dir, dst_dir, args.copies, args.seed, args.workers)

    # Create data.yaml
    create_data_yaml(dst_dir)

    # Summary
    print("\n[augment] SUMMARY:")
    for split in ["train", "val", "test"]:
        img_dir = dst_dir / split / "images"
        n = len(list(img_dir.glob("*"))) if img_dir.exists() else 0
        print(f"  {split}: {n} images")

    total = sum(len(list((dst_dir / s / "images").glob("*"))) for s in ["train", "val", "test"]
                if (dst_dir / s / "images").exists())
    print(f"  TOTAL: {total} images")
    print(f"  Config: {dst_dir}/data.yaml")


if __name__ == "__main__":
    main()
