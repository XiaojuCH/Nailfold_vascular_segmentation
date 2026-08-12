"""Preflight audit for a K2 dataset root before training."""

import argparse
import json
import os
from collections import Counter

import cv2
import numpy as np


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp")


def list_images(directory):
    return sorted(name for name in os.listdir(directory) if name.lower().endswith(IMAGE_EXTENSIONS))


def stems_with_duplicates(filenames):
    counts = Counter(os.path.splitext(filename)[0] for filename in filenames)
    return sorted(stem for stem, count in counts.items() if count > 1)


def main():
    parser = argparse.ArgumentParser(description="Audit K2 train/val/test image-mask alignment.")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    report = {"data_dir": os.path.abspath(args.data_dir), "splits": {}}
    for split in ("train", "val", "test"):
        image_dir = os.path.join(args.data_dir, split, "images")
        mask_dir = os.path.join(args.data_dir, split, "masks")
        if not os.path.isdir(image_dir) or not os.path.isdir(mask_dir):
            raise FileNotFoundError(f"{split} must contain images/ and masks/: {image_dir}, {mask_dir}")
        images = list_images(image_dir)
        masks = list_images(mask_dir)
        if not images or not masks:
            raise ValueError(f"{split} must contain at least one image-mask pair")
        image_set, mask_set = set(images), set(masks)
        missing_masks = sorted(image_set - mask_set)
        orphan_masks = sorted(mask_set - image_set)
        if missing_masks or orphan_masks:
            raise ValueError(
                f"{split} image-mask mismatch: missing_masks={missing_masks[:5]}, orphan_masks={orphan_masks[:5]}"
            )
        duplicate_stems = stems_with_duplicates(images)
        if duplicate_stems:
            raise ValueError(
                f"{split} has duplicate filename stems {duplicate_stems[:5]}. "
                "Soft targets use '<stem>.npy', so each stem must be unique."
            )

        unique_values = set()
        grayscale_masks = 0
        empty_masks = 0
        unreadable_images = []
        image_shapes = Counter()
        mask_shapes = Counter()
        image_mask_shape_mismatches = 0
        for name in images:
            image_path = os.path.join(image_dir, name)
            mask_path = os.path.join(mask_dir, name)
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if image is None:
                unreadable_images.append(image_path)
                continue
            image_shapes[f"{image.shape[1]}x{image.shape[0]}x{image.shape[2]}"] += 1
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise ValueError(f"Cannot read mask: {mask_path}")
            mask_shapes[f"{mask.shape[1]}x{mask.shape[0]}"] += 1
            image_mask_shape_mismatches += int(image.shape[:2] != mask.shape[:2])
            values = np.unique(mask)
            unique_values.update(int(value) for value in values)
            grayscale_masks += int(np.any((mask > 0) & (mask < 255)))
            empty_masks += int(not np.any(mask > 127))
        if unreadable_images:
            raise ValueError(f"Cannot read {len(unreadable_images)} images. First: {unreadable_images[:3]}")
        report["splits"][split] = {
            "images": len(images),
            "masks": len(masks),
            "image_shapes": dict(image_shapes.most_common(10)),
            "mask_shapes": dict(mask_shapes.most_common(10)),
            "image_mask_shape_mismatches": image_mask_shape_mismatches,
            "mask_unique_values_preview": sorted(unique_values)[:30],
            "masks_with_gray_values_1_to_254": grayscale_masks,
            "empty_masks_after_threshold_127": empty_masks,
        }
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.out:
        parent = os.path.dirname(os.path.abspath(args.out))
        os.makedirs(parent, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")


if __name__ == "__main__":
    main()
