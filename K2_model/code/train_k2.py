"""Train the three K2 stages: F0 RGB teacher, F3 green-prior teacher, or K2 student."""

import argparse
import csv
import json
import os
import random
from datetime import datetime

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.dataset_vessel import VesselDataset
from losses.joint_loss import BCEDiceLoss, OutputDistillationLoss
from utils.metrics import average_metric_rows, per_image_metrics_from_logits
from models.green_prior_fusion import GreenPriorFusionModel, PRIOR_FUSION_VARIANTS
from models.transunet_official import TransUNetOfficial


def parse_args():
    parser = argparse.ArgumentParser(description="K2 dual-teacher training.")
    parser.add_argument("--stage", choices=("f0", "f3", "k2"), required=True)
    parser.add_argument("--data_dir", required=True, help="Dataset root containing train/val/test.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--exp_name", default="")
    parser.add_argument("--pretrained", default="", help="Optional R50+ViT-B_16.npz for F0/K2.")
    parser.add_argument("--init_weight", default="", help="Required K2 student initialization: F0 weight.")
    parser.add_argument("--soft_target_dir", default="", help="Required K2 train ensemble_probabilities directory.")
    parser.add_argument("--lambda_kd", type=float, default=1.0)
    parser.add_argument("--f3_variant", choices=PRIOR_FUSION_VARIANTS, default="directional_multiscale")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--intensity_aug", choices=("on", "off"), default="on")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_state_dict(model, path, device, label):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing {label}: {path}")
    try:
        state_dict = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict)
    print(f"[*] Loaded {label}: {path}")


def evaluate(model, loader, device):
    model.eval()
    rows = []
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["image"].to(device))
            rows.extend(per_image_metrics_from_logits(logits, batch["mask"].to(device)))
    return average_metric_rows(rows)


def save_per_image_metrics(path, model, loader, device):
    model.eval()
    rows = []
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["image"].to(device))
            metrics = per_image_metrics_from_logits(logits, batch["mask"].to(device))
            start = len(rows)
            for offset, values in enumerate(metrics):
                rows.append({"filename": loader.dataset.filenames[start + offset], **values})
    with open(path, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    if args.stage == "k2" and not args.init_weight:
        raise ValueError("K2 requires --init_weight from the F0 teacher trained on the same dataset.")
    if args.stage == "k2" and not args.soft_target_dir:
        raise ValueError("K2 requires --soft_target_dir from generate_dual_teacher_targets.py.")
    if args.stage == "k2" and not os.path.isdir(args.soft_target_dir):
        raise FileNotFoundError(f"Missing K2 soft targets: {args.soft_target_dir}")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_name = args.exp_name or f"{args.stage}_seed{args.seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(run_dir, exist_ok=False)

    train_dataset = VesselDataset(
        image_dir=os.path.join(args.data_dir, "train", "images"),
        mask_dir=os.path.join(args.data_dir, "train", "masks"),
        soft_target_dir=args.soft_target_dir if args.stage == "k2" else None,
        img_size=args.img_size,
        augment=True,
        intensity_aug=args.intensity_aug == "on",
    )
    val_dataset = VesselDataset(
        image_dir=os.path.join(args.data_dir, "val", "images"),
        mask_dir=os.path.join(args.data_dir, "val", "masks"),
        img_size=args.img_size,
        augment=False,
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    use_pretrained = args.pretrained if args.stage in ("f0", "k2") else None
    segmentor = TransUNetOfficial(img_size=args.img_size, pretrained_path=use_pretrained)
    if args.stage == "f3":
        model = GreenPriorFusionModel(segmentor, variant=args.f3_variant).to(device)
    else:
        model = segmentor.to(device)

    if args.stage == "k2":
        load_state_dict(model, args.init_weight, device, "F0 student initialization")
        criterion = OutputDistillationLoss(BCEDiceLoss(), lambda_kd=args.lambda_kd).to(device)
    else:
        criterion = BCEDiceLoss().to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)
    config = vars(args).copy()
    config.update({"device": str(device), "run_name": run_name})
    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as handle:
        json.dump(config, handle, ensure_ascii=False, indent=2)

    best_dice = -1.0
    waiting = 0
    log_path = os.path.join(run_dir, "training_log.txt")
    with open(log_path, "w", encoding="utf-8") as log:
        log.write(json.dumps(config, ensure_ascii=False) + "\n")
        for epoch in range(1, args.epochs + 1):
            model.train()
            total_loss = total_seg = total_kd = 0.0
            for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False):
                images = batch["image"].to(device)
                masks = batch["mask"].to(device)
                optimizer.zero_grad()
                logits = model(images)
                if args.stage == "k2":
                    loss, seg_loss, kd_loss = criterion(logits, masks, batch["soft_target"].to(device))
                    total_seg += seg_loss.item()
                    total_kd += kd_loss.item()
                else:
                    loss = criterion(logits, masks)
                    total_seg += loss.item()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            scheduler.step()
            metrics = evaluate(model, val_loader, device)
            batches = len(train_loader)
            kd_text = f" KD:{total_kd / batches:.4f}" if args.stage == "k2" else ""
            line = (
                f"Epoch {epoch:03d} Loss:{total_loss / batches:.4f} Seg:{total_seg / batches:.4f}{kd_text} "
                f"ValDice:{metrics['dice']:.4f} ValHD95:{metrics['hd95']:.2f}"
            )
            print(line)
            log.write(line + "\n")
            log.flush()
            if metrics["dice"] > best_dice:
                best_dice = metrics["dice"]
                waiting = 0
                torch.save(model.state_dict(), os.path.join(run_dir, "best_model.pth"))
            else:
                waiting += 1
                if waiting >= args.patience:
                    print(f"[*] Early stopping after {args.patience} epochs without validation improvement.")
                    break

    load_state_dict(model, os.path.join(run_dir, "best_model.pth"), device, "best validation checkpoint")
    save_per_image_metrics(os.path.join(run_dir, "val_per_image.csv"), model, val_loader, device)
    print(f"[K2 stage complete] {args.stage}: {run_dir}")


if __name__ == "__main__":
    main()
