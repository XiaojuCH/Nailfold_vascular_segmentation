"""Unified K2 evaluator with per-image and aggregate segmentation metrics."""

import argparse
import csv
import json
import os
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.dataset_vessel import VesselDataset
from utils.metrics import METRIC_KEYS, average_metric_rows, per_image_metrics_from_logits
from models.transunet_official import TransUNetOfficial


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a K2/F0 TransUNet checkpoint.")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--weight", required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--name", default="K2")
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--boundary_tolerance", type=int, default=2)
    return parser.parse_args()


def load_state_dict(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def main():
    args = parse_args()
    if not os.path.isfile(args.weight):
        raise FileNotFoundError(f"Missing checkpoint: {args.weight}")
    if not 0.0 < args.threshold < 1.0:
        raise ValueError("--threshold must be in (0, 1)")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = VesselDataset(
        image_dir=os.path.join(args.data_dir, args.split, "images"),
        mask_dir=os.path.join(args.data_dir, args.split, "masks"),
        img_size=args.img_size,
        augment=False,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    model = TransUNetOfficial(img_size=args.img_size).to(device)
    model.load_state_dict(load_state_dict(args.weight, device))
    model.eval()

    rows = []
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Evaluate {args.name}"):
            logits = model(batch["image"].to(device))
            metrics = per_image_metrics_from_logits(
                logits,
                batch["mask"].to(device),
                threshold=args.threshold,
                boundary_tolerance=args.boundary_tolerance,
            )
            start = len(rows)
            for offset, values in enumerate(metrics):
                rows.append({"filename": dataset.filenames[start + offset], **values})

    run_dir = os.path.join(args.out_dir, datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(run_dir, exist_ok=False)
    per_image_path = os.path.join(run_dir, "per_image_metrics.csv")
    with open(per_image_path, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=["filename", *METRIC_KEYS, "tp", "tn", "fp", "fn"])
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "name": args.name,
        "weight": os.path.abspath(args.weight),
        "data_dir": os.path.abspath(args.data_dir),
        "split": args.split,
        "n_images": len(rows),
        "img_size": args.img_size,
        "threshold": args.threshold,
        "boundary_tolerance": args.boundary_tolerance,
        **average_metric_rows(rows),
        "per_image_csv": per_image_path,
    }
    with open(os.path.join(run_dir, "aggregate_metrics.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    with open(os.path.join(run_dir, "aggregate_metrics.csv"), "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary.keys())
        writer.writeheader()
        writer.writerow(summary)
    print("[K2 evaluation complete]")
    print(
        f"{args.name}: Dice={summary['dice']:.4f}, IoU={summary['iou']:.4f}, "
        f"Recall={summary['sensitivity']:.4f}, Precision={summary['precision']:.4f}, "
        f"HD95={summary['hd95']:.2f}, clDice={summary['cldice']:.4f}, "
        f"BoundaryF1={summary['boundary_f1']:.4f}"
    )
    print(f"Output: {run_dir}")


if __name__ == "__main__":
    main()
