"""Prepare a non-destructive MMSegmentation view of dataset_all_filtered.

Images are hard-linked when supported, while masks are re-encoded to exact 0/1
classes using the project's existing threshold (>127). This prevents MMSeg from
interpreting anti-aliased 0..255 mask pixels as hundreds of class labels.
"""

import argparse
import os
from pathlib import Path

import cv2
import numpy as np


SPLITS = ("train", "val", "test")


def parse_args():
    parser = argparse.ArgumentParser(description="Create an MMSeg-compatible binary dataset view.")
    parser.add_argument("--source", default="dataset_all_filtered")
    parser.add_argument("--output", default="dataset_all_filtered_mmseg")
    parser.add_argument("--threshold", type=int, default=127)
    parser.add_argument("--overwrite_masks", action="store_true")
    return parser.parse_args()


def ensure_image_link(source: Path, destination: Path):
    if destination.exists():
        if destination.stat().st_size != source.stat().st_size:
            raise ValueError(f"Existing image link has different size: {destination}")
        return
    try:
        os.link(source, destination)
    except OSError as error:
        raise RuntimeError(
            f"Cannot create hard link for {source}. The source/output must be on the same drive. "
            "Choose an output folder on D: or create the dataset view manually."
        ) from error


def write_binary_mask(source: Path, destination: Path, threshold: int, overwrite: bool):
    if destination.exists() and not overwrite:
        existing = cv2.imread(str(destination), cv2.IMREAD_GRAYSCALE)
        if existing is not None and set(np.unique(existing).tolist()).issubset({0, 1}):
            return
    mask = cv2.imread(str(source), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Cannot read mask: {source}")
    binary = (mask > threshold).astype(np.uint8)
    if not cv2.imwrite(str(destination), binary):
        raise RuntimeError(f"Cannot write mask: {destination}")


def main():
    args = parse_args()
    source_root = Path(args.source).resolve()
    output_root = Path(args.output).resolve()
    if source_root == output_root:
        raise ValueError("Output directory must differ from source directory")

    summary = []
    for split in SPLITS:
        source_images = source_root / split / "images"
        source_masks = source_root / split / "masks"
        output_images = output_root / split / "images"
        output_masks = output_root / split / "masks"
        if not source_images.is_dir() or not source_masks.is_dir():
            raise FileNotFoundError(f"Missing source split folders for {split}")
        output_images.mkdir(parents=True, exist_ok=True)
        output_masks.mkdir(parents=True, exist_ok=True)

        images = sorted(path for path in source_images.iterdir() if path.suffix.lower() in {".png", ".jpg", ".jpeg"})
        missing_masks = [path.name for path in images if not (source_masks / path.name).is_file()]
        if missing_masks:
            raise FileNotFoundError(f"{split} missing {len(missing_masks)} masks, first: {missing_masks[:5]}")
        for image_path in images:
            ensure_image_link(image_path, output_images / image_path.name)
            write_binary_mask(
                source_masks / image_path.name,
                output_masks / image_path.name,
                args.threshold,
                args.overwrite_masks,
            )
        output_masks_list = sorted(output_masks.glob("*.png"))
        if len(images) != len(output_masks_list):
            raise RuntimeError(f"{split} count mismatch: images={len(images)}, masks={len(output_masks_list)}")
        unique = set()
        for mask_path in output_masks_list:
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            unique.update(np.unique(mask).tolist())
        if not unique.issubset({0, 1}):
            raise ValueError(f"{split} contains non-binary labels: {sorted(unique)}")
        summary.append((split, len(images), sorted(unique)))

    print("[MMSeg dataset view complete]")
    for split, count, unique in summary:
        print(f"{split}: images={count}, masks={count}, labels={unique}")
    print(f"output: {output_root}")


if __name__ == "__main__":
    main()
