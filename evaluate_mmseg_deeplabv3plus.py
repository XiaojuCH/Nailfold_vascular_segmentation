"""Evaluate an official MMSegmentation checkpoint with this project's metrics."""

import argparse
import csv
import os
import sys

import cv2
import numpy as np
import torch
from tqdm import tqdm

from utils.metrics import METRIC_KEYS, average_metric_rows, binary_metrics_from_masks


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate MMSeg DeepLabV3+ with unified nailfold metrics.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_dir", default="./dataset_all_filtered")
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def write_csv(path, rows, fields):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    if not os.path.isfile(args.config):
        raise FileNotFoundError(f"Missing config: {args.config}")
    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(f"Missing checkpoint: {args.checkpoint}")

    mmseg_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "TYT_Code", "mmsegmentation-main")
    if not os.path.isdir(mmseg_root):
        raise FileNotFoundError(f"Missing MMSeg source tree: {mmseg_root}")
    sys.path.insert(0, mmseg_root)
    from mmseg.apis import inference_model, init_model

    image_dir = os.path.join(args.data_dir, args.split, "images")
    mask_dir = os.path.join(args.data_dir, args.split, "masks")
    filenames = sorted(name for name in os.listdir(image_dir) if name.lower().endswith(".png"))
    if not filenames:
        raise FileNotFoundError(f"No PNG images found: {image_dir}")
    missing = [name for name in filenames if not os.path.isfile(os.path.join(mask_dir, name))]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} masks, first: {missing[:5]}")

    model = init_model(args.config, args.checkpoint, device=args.device)
    model.eval()
    prediction_dir = os.path.join(args.out_dir, "predictions")
    os.makedirs(prediction_dir, exist_ok=True)
    rows = []
    with torch.no_grad():
        for filename in tqdm(filenames, desc="MMSeg unified evaluation"):
            result = inference_model(model, os.path.join(image_dir, filename))
            prediction = result.pred_sem_seg.data.squeeze().detach().cpu().numpy().astype(bool)
            target_image = cv2.imread(os.path.join(mask_dir, filename), cv2.IMREAD_GRAYSCALE)
            if target_image is None:
                raise FileNotFoundError(filename)
            target = cv2.resize(target_image, prediction.shape[::-1], interpolation=cv2.INTER_NEAREST) > 127
            metrics = binary_metrics_from_masks(prediction, target, boundary_tolerance=2)
            if not cv2.imwrite(os.path.join(prediction_dir, filename), prediction.astype(np.uint8) * 255):
                raise RuntimeError(f"Cannot save prediction: {filename}")
            rows.append({"filename": filename, **{key: metrics[key] for key in METRIC_KEYS}, **{key: metrics[key] for key in ("tp", "tn", "fp", "fn")}})

    aggregate = average_metric_rows(rows)
    write_csv(os.path.join(args.out_dir, "per_image_metrics.csv"), rows, ["filename", *METRIC_KEYS, "tp", "tn", "fp", "fn"])
    write_csv(
        os.path.join(args.out_dir, "aggregate_metrics.csv"),
        [{"model": "MMSegmentation DeepLabV3+", "checkpoint": args.checkpoint, "n_images": len(rows), **aggregate}],
        ["model", "checkpoint", "n_images", *METRIC_KEYS],
    )
    print("[MMSeg unified evaluation complete]")
    print(f"n_images={len(rows)} Dice={aggregate['dice']:.4f} HD95={aggregate['hd95']:.2f} clDice={aggregate['cldice']:.4f} BoundaryF1={aggregate['boundary_f1']:.4f}")
    print(f"out_dir={args.out_dir}")


if __name__ == "__main__":
    main()
