"""
Train strong encoder-decoder baselines from segmentation_models_pytorch.

This is a probe for whether the current TransUNet backbone is the main bottleneck.
It uses the same VesselDataset and metric implementation as evaluate_all.py.
"""
import argparse
import csv
import json
import os
import random
import sys
from datetime import datetime

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datasets.dataset_vessel import VesselDataset
from losses.joint_loss import build_segmentation_loss
from utils.metrics import METRIC_KEYS, average_metric_rows, per_image_metrics_from_logits

try:
    import segmentation_models_pytorch as smp
except ImportError as exc:
    raise ImportError(
        "segmentation_models_pytorch is required for train_smp_baseline.py. "
        "Install it in the active environment or run another baseline script."
    ) from exc


DATASETS = {
    "jiabi": "./dataset_raw_split",
    "anfc256": "./dataset_anfc256_split",
    "all": "./dataset_all_split",
    "all_filtered": "./dataset_all_filtered",
    "all_filtered_VT_Turn": "./dataset_all_filtered_VT_Turn",
}

ARCHES = ["unet", "unetplusplus", "fpn", "deeplabv3plus", "linknet", "pspnet"]
SEG_LOSSES = [
    "bce_dice",
    "focal_tversky",
    "unified_focal",
    "bce_dice_cldice",
    "bce_dice_boundary",
    "bce_dice_cldice_boundary",
    "bce_dice_cbdice",
    "bce_dice_cbdice_boundary",
]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def safe_name(text):
    out = []
    for ch in text:
        if ch.isalnum():
            out.append(ch)
        else:
            out.append("_")
    name = "".join(out).strip("_")
    while "__" in name:
        name = name.replace("__", "_")
    return name or "run"


def get_args():
    parser = argparse.ArgumentParser(description="Train SMP strong segmentation baselines.")
    parser.add_argument("--arch", default="deeplabv3plus", choices=ARCHES)
    parser.add_argument("--encoder_name", default="resnet34")
    parser.add_argument("--encoder_weights", default="", help="Use empty string for random init, or e.g. imagenet.")
    parser.add_argument("--dataset", default="all_filtered", choices=list(DATASETS.keys()))
    parser.add_argument("--data_dir", default="")
    parser.add_argument("--save_dir", default="./results/experiments")
    parser.add_argument("--exp_name", default="")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--boundary_tolerance", type=int, default=2)
    parser.add_argument("--seg_loss", default="bce_dice", choices=SEG_LOSSES)
    parser.add_argument("--cldice_weight", type=float, default=0.5)
    parser.add_argument("--boundary_weight", type=float, default=0.5)
    parser.add_argument("--cbdice_weight", type=float, default=0.5)
    parser.add_argument("--focal_alpha", type=float, default=0.3)
    parser.add_argument("--focal_beta", type=float, default=0.7)
    parser.add_argument("--focal_gamma", type=float, default=0.75)
    return parser.parse_args()


def build_smp_model(args):
    encoder_weights = args.encoder_weights if args.encoder_weights else None
    kwargs = {
        "encoder_name": args.encoder_name,
        "encoder_weights": encoder_weights,
        "in_channels": 3,
        "classes": 1,
        "activation": None,
    }
    if args.arch == "unet":
        return smp.Unet(**kwargs)
    if args.arch == "unetplusplus":
        return smp.UnetPlusPlus(**kwargs)
    if args.arch == "fpn":
        return smp.FPN(**kwargs)
    if args.arch == "deeplabv3plus":
        return smp.DeepLabV3Plus(**kwargs)
    if args.arch == "linknet":
        return smp.Linknet(**kwargs)
    if args.arch == "pspnet":
        return smp.PSPNet(**kwargs)
    raise ValueError(f"Unknown arch: {args.arch}")


def write_csv(path, rows, fieldnames):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def evaluate(model, dataset, loader, args, device, run_dir):
    model.eval()
    rows = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Test"):
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            logits = model(images)
            batch_rows = per_image_metrics_from_logits(
                logits,
                masks,
                threshold=args.threshold,
                boundary_tolerance=args.boundary_tolerance,
            )
            start = len(rows)
            for offset, metrics in enumerate(batch_rows):
                rows.append({
                    "experiment": args.exp_name,
                    "model_type": "smp",
                    "filename": dataset.filenames[start + offset],
                    **{key: metrics[key] for key in METRIC_KEYS},
                    "tp": metrics["tp"],
                    "tn": metrics["tn"],
                    "fp": metrics["fp"],
                    "fn": metrics["fn"],
                })

    per_image_path = os.path.join(run_dir, "per_image_results.csv")
    write_csv(per_image_path, rows, ["experiment", "model_type", "filename", *METRIC_KEYS, "tp", "tn", "fp", "fn"])
    avg = average_metric_rows(rows)
    summary = {
        "experiment": args.exp_name,
        "model_type": "smp",
        "arch": args.arch,
        "encoder_name": args.encoder_name,
        "encoder_weights": args.encoder_weights,
        "dataset": args.data_dir,
        "split": "test",
        "threshold": args.threshold,
        "img_size": args.img_size,
        "seed": args.seed,
        "seg_loss": args.seg_loss,
        "cldice_weight": args.cldice_weight,
        "boundary_weight": args.boundary_weight,
        "cbdice_weight": args.cbdice_weight,
        "n_images": len(rows),
        **avg,
        "per_image_csv": per_image_path,
    }
    aggregate_path = os.path.join(run_dir, "aggregate_results.csv")
    write_csv(aggregate_path, [summary], list(summary.keys()))
    return summary, aggregate_path


def main():
    args = get_args()
    set_seed(args.seed)
    if not args.data_dir:
        args.data_dir = DATASETS[args.dataset]
    if not args.exp_name:
        weights_tag = args.encoder_weights if args.encoder_weights else "scratch"
        args.exp_name = f"{args.dataset}/smp_{args.arch}_{safe_name(args.encoder_name)}_{safe_name(weights_tag)}_seed{args.seed}"

    timestamp = datetime.now().strftime("%m%d_%H%M")
    run_dir = os.path.join(args.save_dir, args.exp_name, timestamp)
    os.makedirs(run_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] SMP baseline: {args.arch}, encoder={args.encoder_name}, weights={args.encoder_weights or 'none'}")
    print(f"[*] dataset: {args.dataset} ({args.data_dir})")
    print(f"[*] save_path: {run_dir}")
    print(f"[*] device: {device}, seed={args.seed}, seg_loss={args.seg_loss}")

    train_dataset = VesselDataset(
        image_dir=os.path.join(args.data_dir, "train/images"),
        mask_dir=os.path.join(args.data_dir, "train/masks"),
        teacher_dir=None,
        img_size=args.img_size,
        augment=True,
    )
    val_dataset = VesselDataset(
        image_dir=os.path.join(args.data_dir, "val/images"),
        mask_dir=os.path.join(args.data_dir, "val/masks"),
        teacher_dir=None,
        img_size=args.img_size,
        augment=False,
    )
    test_dataset = VesselDataset(
        image_dir=os.path.join(args.data_dir, "test/images"),
        mask_dir=os.path.join(args.data_dir, "test/masks"),
        teacher_dir=None,
        img_size=args.img_size,
        augment=False,
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = build_smp_model(args).to(device)
    criterion = build_segmentation_loss(
        seg_loss=args.seg_loss,
        cldice_weight=args.cldice_weight,
        boundary_weight=args.boundary_weight,
        cbdice_weight=args.cbdice_weight,
        focal_alpha=args.focal_alpha,
        focal_beta=args.focal_beta,
        focal_gamma=args.focal_gamma,
    ).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    config = vars(args).copy()
    config["device"] = device
    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    log_path = os.path.join(run_dir, "training_log.txt")
    best_dice = -1.0
    patience_counter = 0
    with open(log_path, "w", encoding="utf-8") as log_file:
        for epoch in range(args.epochs):
            model.train()
            train_loss = 0.0
            for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs} [Train]", leave=False):
                images = batch["image"].to(device)
                masks = batch["mask"].to(device)
                optimizer.zero_grad()
                logits = model(images)
                loss = criterion(logits, masks)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            scheduler.step()

            model.eval()
            val_rows = []
            val_loss = 0.0
            with torch.no_grad():
                for batch in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{args.epochs} [Val]", leave=False):
                    images = batch["image"].to(device)
                    masks = batch["mask"].to(device)
                    logits = model(images)
                    val_loss += criterion(logits, masks).item()
                    val_rows.extend(per_image_metrics_from_logits(logits, masks, threshold=args.threshold))
            val_avg = average_metric_rows(val_rows)
            line = (
                f"Epoch {epoch + 1:03d} | LR={scheduler.get_last_lr()[0]:.6f} | "
                f"TrainLoss={train_loss / max(len(train_loader), 1):.4f} | "
                f"ValLoss={val_loss / max(len(val_loader), 1):.4f} | "
                f"Dice={val_avg['dice']:.4f} | IoU={val_avg['iou']:.4f} | "
                f"Recall={val_avg['sensitivity']:.4f} | Precision={val_avg['precision']:.4f} | "
                f"HD95={val_avg['hd95']:.2f} | clDice={val_avg['cldice']:.4f} | BoundaryF1={val_avg['boundary_f1']:.4f}"
            )
            print(line)
            log_file.write(line + "\n")
            log_file.flush()

            if val_avg["dice"] > best_dice:
                best_dice = val_avg["dice"]
                patience_counter = 0
                torch.save(model.state_dict(), os.path.join(run_dir, "best_model.pth"))
                print(f"[*] Saved best model: val Dice={best_dice:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print(f"[*] Early stopping at epoch {epoch + 1}")
                    break

    best_path = os.path.join(run_dir, "best_model.pth")
    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    summary, aggregate_path = evaluate(model, test_dataset, test_loader, args, device, run_dir)
    print("\n[SMP Evaluation Complete]")
    print(f"Aggregate CSV: {aggregate_path}")
    print(
        f"{summary['experiment']}: Dice={summary['dice']:.4f}, IoU={summary['iou']:.4f}, "
        f"Recall={summary['sensitivity']:.4f}, Precision={summary['precision']:.4f}, "
        f"HD95={summary['hd95']:.2f}, clDice={summary['cldice']:.4f}, BoundaryF1={summary['boundary_f1']:.4f}"
    )


if __name__ == "__main__":
    main()

