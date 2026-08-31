#!/usr/bin/env python3
"""Download and prepare fire/smoke dataset for YOLOv8 training.

Downloads the D-Fire dataset from OneDrive (official source) or
provides instructions for manual download from Kaggle.

Classes: 0 = smoke, 1 = fire
"""

import argparse
import os
import random
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from urllib.request import urlretrieve


def parse_args():
    p = argparse.ArgumentParser(description="Download and prepare D-Fire dataset")
    p.add_argument("--out", default="data/dfire", help="output directory")
    p.add_argument("--val-split", type=float, default=0.15, help="fraction for validation (from train)")
    p.add_argument("--seed", type=int, default=42, help="random seed for split")
    p.add_argument("--source", default="kaggle",
                   choices=["kaggle", "manual"],
                   help="download source")
    return p.parse_args()


def download_kaggle(raw_dir: Path) -> bool:
    """Download from Kaggle using kaggle CLI."""
    dataset_slug = "sayedgamal99/smoke-fire-detection-yolo"
    print(f"[download] Downloading from Kaggle: {dataset_slug} ...")

    try:
        import kaggle  # noqa: F401
    except ImportError:
        print("[download] Install kaggle: pip install kaggle")
        return False

    # Check credentials
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_json.exists():
        print("[download] Kaggle credentials not found!")
        print("  1. Go to https://www.kaggle.com/settings → API → Create New Token")
        print(f"  2. Save the downloaded kaggle.json to {kaggle_json}")
        print(f"  3. chmod 600 {kaggle_json}")
        print("  4. Re-run this script")
        return False

    raw_dir.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", dataset_slug,
             "-p", str(raw_dir), "--unzip"],
            check=True,
            timeout=1200,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"[download] Kaggle download failed: {e}")
        return False


def find_dataset_dirs(raw_dir: Path):
    """Auto-discover the dataset structure."""
    # Look for train/val/test dirs with images/labels
    splits = {}
    for split_name in ["train", "val", "valid", "test"]:
        for d in raw_dir.rglob(split_name):
            if d.is_dir():
                img_dir = d / "images"
                lbl_dir = d / "labels"
                if img_dir.exists() and lbl_dir.exists():
                    canonical = "val" if split_name == "valid" else split_name
                    if canonical not in splits:
                        splits[canonical] = {"images": img_dir, "labels": lbl_dir}
                        n_imgs = len(list(img_dir.glob("*")))
                        print(f"  Found {split_name}: {n_imgs} images at {img_dir}")

    return splits


def copy_split(src_img_dir: Path, src_lbl_dir: Path,
               dst_img_dir: Path, dst_lbl_dir: Path, files: list = None):
    """Copy images and labels to destination dirs."""
    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_lbl_dir.mkdir(parents=True, exist_ok=True)

    if files is None:
        files = sorted(
            list(src_img_dir.glob("*.jpg")) +
            list(src_img_dir.glob("*.jpeg")) +
            list(src_img_dir.glob("*.png"))
        )

    copied = 0
    for img_file in files:
        shutil.copy2(img_file, dst_img_dir / img_file.name)
        label_name = img_file.stem + ".txt"
        src_label = src_lbl_dir / label_name
        if src_label.exists():
            shutil.copy2(src_label, dst_lbl_dir / label_name)
        copied += 1

    return copied


def create_data_yaml(out_dir: Path):
    """Create data.yaml for Ultralytics."""
    yaml_content = f"""# D-Fire dataset for YOLOv8 fire/smoke detection
# Classes: 0 = smoke, 1 = fire

path: {out_dir.resolve()}
train: train/images
val: val/images
test: test/images

names:
  0: smoke
  1: fire
"""
    yaml_path = out_dir / "data.yaml"
    yaml_path.write_text(yaml_content)
    print(f"[download] Created {yaml_path}")


def main():
    args = parse_args()
    out_dir = Path(args.out)
    raw_dir = Path("data/dfire_raw")

    # Check if already prepared
    if (out_dir / "data.yaml").exists():
        print(f"[download] Dataset already exists at {out_dir}/")
        for split in ["train", "val", "test"]:
            d = out_dir / split / "images"
            n = len(list(d.glob("*"))) if d.exists() else 0
            print(f"  {split}: {n}")
        print("[download] To re-download, delete the directory first.")
        return

    # Download
    success = False
    if args.source == "kaggle":
        success = download_kaggle(raw_dir)

    if not success:
        print("\n" + "=" * 60)
        print("MANUAL DOWNLOAD INSTRUCTIONS")
        print("=" * 60)
        print()
        print("Option A - Kaggle (recommended):")
        print("  1. Go to: https://www.kaggle.com/datasets/sayedgamal99/smoke-fire-detection-yolo")
        print("  2. Click 'Download' (need free Kaggle account)")
        print(f"  3. Extract the ZIP to: {raw_dir}/")
        print(f"  4. Re-run: python {__file__} --source manual")
        print()
        print("Option B - OneDrive (official D-Fire):")
        print("  1. Go to: https://1drv.ms/f/c/c0bd25b6b048b01d/Ema8FFze8mFIlM1Hn81BUUgBE3vnnmK4SQxybS-nHRt2pA")
        print("  2. Download train.zip, val.zip, test.zip")
        print(f"  3. Extract all to: {raw_dir}/")
        print(f"  4. Re-run: python {__file__} --source manual")
        print()
        print(f"Expected structure after extraction:")
        print(f"  {raw_dir}/")
        print(f"    train/")
        print(f"      images/  (*.jpg)")
        print(f"      labels/  (*.txt)")
        print(f"    val/ or valid/")
        print(f"      images/")
        print(f"      labels/")
        print(f"    test/")
        print(f"      images/")
        print(f"      labels/")

        if args.source != "manual":
            sys.exit(1)

    if not raw_dir.exists():
        print(f"[download] ERROR: Raw data directory not found: {raw_dir}")
        print("  Please download the dataset first (see instructions above)")
        sys.exit(1)

    # Discover structure
    print("[download] Scanning dataset structure ...")
    splits = find_dataset_dirs(raw_dir)

    if not splits:
        print("[download] ERROR: Could not find train/val/test splits!")
        print(f"  Searched in: {raw_dir}")
        print("  Expected: subdirectories with images/ and labels/ folders")
        sys.exit(1)

    # Copy to output
    out_dir.mkdir(parents=True, exist_ok=True)

    if "train" in splits:
        src = splits["train"]
        if "val" not in splits:
            # Create val split from train
            all_imgs = sorted(
                list(src["images"].glob("*.jpg")) +
                list(src["images"].glob("*.jpeg")) +
                list(src["images"].glob("*.png"))
            )
            random.seed(args.seed)
            random.shuffle(all_imgs)
            n_val = int(len(all_imgs) * args.val_split)
            val_imgs = all_imgs[:n_val]
            train_imgs = all_imgs[n_val:]

            print(f"[download] Splitting train: {len(train_imgs)} train, {n_val} val")
            n = copy_split(src["images"], src["labels"],
                          out_dir / "train/images", out_dir / "train/labels", train_imgs)
            print(f"  train: {n} files")
            n = copy_split(src["images"], src["labels"],
                          out_dir / "val/images", out_dir / "val/labels", val_imgs)
            print(f"  val: {n} files")
        else:
            print("[download] Copying train ...")
            n = copy_split(src["images"], src["labels"],
                          out_dir / "train/images", out_dir / "train/labels")
            print(f"  train: {n} files")

    if "val" in splits:
        print("[download] Copying val ...")
        src = splits["val"]
        n = copy_split(src["images"], src["labels"],
                      out_dir / "val/images", out_dir / "val/labels")
        print(f"  val: {n} files")

    if "test" in splits:
        print("[download] Copying test ...")
        src = splits["test"]
        n = copy_split(src["images"], src["labels"],
                      out_dir / "test/images", out_dir / "test/labels")
        print(f"  test: {n} files")
    elif "val" in splits or (out_dir / "val/images").exists():
        print("[download] No test split, duplicating val as test")
        shutil.copytree(out_dir / "val", out_dir / "test", dirs_exist_ok=True)

    # Create data.yaml
    create_data_yaml(out_dir)

    # Cleanup raw
    if raw_dir.exists():
        print("[download] Cleaning up raw download ...")
        shutil.rmtree(raw_dir)

    # Summary
    print(f"\n[download] DONE!")
    for split in ["train", "val", "test"]:
        d = out_dir / split / "images"
        n = len(list(d.glob("*"))) if d.exists() else 0
        print(f"  {split}: {n} images")
    print(f"  config: {out_dir}/data.yaml")


if __name__ == "__main__":
    main()
