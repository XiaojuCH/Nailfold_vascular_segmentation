"""
Train green-prior joint models with segmentation_models_pytorch segmentors.

This probes whether the current gain ceiling comes from the TransUNet backbone.
The model keeps the existing Enhancer + teacher-prior distillation design, but
replaces the segmentor with a stronger SMP encoder-decoder.
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
from losses.joint_loss import JointDistillationBoundaryLoss, JointDistillationLoss
from models.joint_framework import (
    Enhancer,
    JointModel,
    JointModel_BoundaryRefine,
    JointModel_Gated,
    JointModel_V2,
    MultiScaleEnhancer,
)
from train_smp_baseline import ARCHES, build_smp_model, safe_name
from utils.metrics import METRIC_KEYS, average_metric_rows, per_image_metrics_from_logits


DATASETS = {
    "jiabi": "./dataset_raw_split",
    "anfc256": "./dataset_anfc256_split",
    "all": "./dataset_all_split",
    "all_filtered": "./dataset_all_filtered",
    "all_filtered_VT_Turn": "./dataset_all_filtered_VT_Turn",
}

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

TEACHER_MODES = [
    "green+clahe",
    "clahe_only",
    "green_only",
    "green_blackhat",
    "green_clahe_blackhat",
    "green_frangi",
]

JOINT_MODELS = ["v1", "v2", "gated", "boundary_refine"]
ENHANCERS = ["basic", "multiscale"]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_teacher_dir(data_dir, teacher_mode):
    if teacher_mode == "green+clahe" and os.path.isdir(os.path.join(data_dir, "train", "teacher_priors")):
        return os.path.join(data_dir, "train", "teacher_priors")
    teacher_suffix = teacher_mode.replace("+", "_")
    return os.path.join(data_dir, "train", f"teacher_priors_{teacher_suffix}")


def write_csv(path, rows, fieldnames):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_joint_model(args):
    segmentor = build_smp_model(args)
    enhancer = (
        MultiScaleEnhancer(in_channels=3, out_channels=3)
        if args.enhancer == "multiscale"
        else Enhancer(in_channels=3, out_channels=3)
    )
    if args.joint_model == "v2":
        return JointModel_V2(enhancer, segmentor, attention_mode=args.attention_mode)
    if args.joint_model == "gated":
        return JointModel_Gated(enhancer, segmentor)
    if args.joint_model == "boundary_refine":
        return JointModel_BoundaryRefine(enhancer, segmentor)
    return JointModel(enhancer, segmentor)


def forward_joint(model, images, joint_model):
    if joint_model == "boundary_refine":
        seg_logits, enhanced, boundary_logits = model(images)
        return seg_logits, enhanced, boundary_logits
    if joint_model in ["v2", "gated"]:
        seg_logits, enhanced, aux = model(images)
        return seg_logits, enhanced, aux
    seg_logits, enhanced = model(images)
    return seg_logits, enhanced, None


def get_args():
    parser = argparse.ArgumentParser(description="Train SMP green-prior joint models.")
    parser.add_argument("--arch", default="fpn", choices=ARCHES)
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

    parser.add_argument("--teacher_mode", default="green_only", choices=TEACHER_MODES)
    parser.add_argument("--enhancer", default="basic", choices=ENHANCERS)
    parser.add_argument("--joint_model", default="v1", choices=JOINT_MODELS)
    parser.add_argument("--attention_mode", default="normal", choices=["normal", "inverse"])
    parser.add_argument("--loss_weighting", default="fixed", choices=["fixed", "learnable"])
    parser.add_argument("--lambda_mse", type=float, default=10.0)
    parser.add_argument("--lambda_grad", type=float, default=0.0)
    parser.add_argument("--boundary_aux_weight", type=float, default=0.3)

    parser.add_argument("--seg_loss", default="bce_dice", choices=SEG_LOSSES)
    parser.add_argument("--cldice_weight", type=float, default=0.5)
    parser.add_argument("--boundary_weight", type=float, default=0.5)
    parser.add_argument("--cbdice_weight", type=float, default=0.5)
    parser.add_argument("--focal_alpha", type=float, default=0.3)
    parser.add_argument("--focal_beta", type=float, default=0.7)
    parser.add_argument("--focal_gamma", type=float, default=0.75)
    return parser.parse_args()


def evaluate(model, dataset, loader, args, device, run_dir):
    model.eval()
    rows = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Test"):
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            logits, _, _ = forward_joint(model, images, args.joint_model)
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
                    "model_type": "smp_joint",
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
        "model_type": "smp_joint",
        "arch": args.arch,
        "encoder_name": args.encoder_name,
        "encoder_weights": args.encoder_weights,
        "dataset": args.data_dir,
        "split": "test",
        "threshold": args.threshold,
        "img_size": args.img_size,
        "seed": args.seed,
        "teacher_mode": args.teacher_mode,
        "enhancer": args.enhancer,
        "joint_model": args.joint_model,
        "attention_mode": args.attention_mode,
        "loss_weighting": args.loss_weighting,
        "lambda_mse": args.lambda_mse,
        "lambda_grad": args.lambda_grad,
        "seg_loss": args.seg_loss,
        "cldice_weight": args.cldice_weight,
        "boundary_weight": args.boundary_weight,
        "cbdice_weight": args.cbdice_weight,
        "boundary_aux_weight": args.boundary_aux_weight,
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
        teacher_tag = args.teacher_mode.replace("+", "_")
        args.exp_name = (
            f"{args.dataset}/smp_joint_{args.arch}_{safe_name(args.encoder_name)}_"
            f"{safe_name(weights_tag)}_{teacher_tag}_{args.seg_loss}_seed{args.seed}"
        )

    timestamp = datetime.now().strftime("%m%d_%H%M")
    run_dir = os.path.join(args.save_dir, args.exp_name, timestamp)
    os.makedirs(run_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    teacher_dir = get_teacher_dir(args.data_dir, args.teacher_mode)
    if not os.path.isdir(teacher_dir):
        raise FileNotFoundError(f"Missing teacher prior directory: {teacher_dir}")

    print(f"[*] SMP joint: {args.arch}, encoder={args.encoder_name}, weights={args.encoder_weights or 'none'}")
    print(f"[*] dataset: {args.dataset} ({args.data_dir})")
    print(f"[*] teacher_mode: {args.teacher_mode} ({teacher_dir})")
    print(f"[*] enhancer={args.enhancer}, joint_model={args.joint_model}, seg_loss={args.seg_loss}")
    print(f"[*] distill: lambda_mse={args.lambda_mse}, lambda_grad={args.lambda_grad}, loss_weighting={args.loss_weighting}")
    print(f"[*] save_path: {run_dir}")
    print(f"[*] device: {device}, seed={args.seed}")

    train_dataset = VesselDataset(
        image_dir=os.path.join(args.data_dir, "train/images"),
        mask_dir=os.path.join(args.data_dir, "train/masks"),
        teacher_dir=teacher_dir,
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

    model = build_joint_model(args).to(device)
    criterion_class = JointDistillationBoundaryLoss if args.joint_model == "boundary_refine" else JointDistillationLoss
    criterion_kwargs = {"boundary_aux_weight": args.boundary_aux_weight} if args.joint_model == "boundary_refine" else {}
    criterion = criterion_class(
        lambda_mse=args.lambda_mse,
        lambda_grad=args.lambda_grad,
        weight_mode=args.loss_weighting,
        seg_loss=args.seg_loss,
        cldice_weight=args.cldice_weight,
        boundary_weight=args.boundary_weight,
        cbdice_weight=args.cbdice_weight,
        focal_alpha=args.focal_alpha,
        focal_beta=args.focal_beta,
        focal_gamma=args.focal_gamma,
        **criterion_kwargs,
    ).to(device)

    optim_params = list(model.parameters())
    if args.loss_weighting == "learnable":
        optim_params += list(criterion.parameters())
    optimizer = optim.AdamW(optim_params, lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    config = vars(args).copy()
    config["device"] = device
    config["teacher_dir"] = teacher_dir
    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    log_path = os.path.join(run_dir, "training_log.txt")
    best_dice = -1.0
    patience_counter = 0
    with open(log_path, "w", encoding="utf-8") as log_file:
        for epoch in range(args.epochs):
            model.train()
            train_loss = 0.0
            train_seg = 0.0
            train_mse = 0.0
            train_grad = 0.0
            train_boundary = 0.0

            for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs} [Train]", leave=False):
                images = batch["image"].to(device)
                masks = batch["mask"].to(device)
                teachers = batch["teacher"].to(device)
                optimizer.zero_grad()

                seg_logits, enhanced, aux = forward_joint(model, images, args.joint_model)
                if args.joint_model == "boundary_refine":
                    loss, l_seg, l_mse, l_grad, l_boundary = criterion(seg_logits, masks, enhanced, teachers, aux)
                    train_boundary += l_boundary.item()
                else:
                    loss, l_seg, l_mse, l_grad = criterion(seg_logits, masks, enhanced, teachers)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                train_seg += l_seg.item()
                train_mse += l_mse.item()
                train_grad += l_grad.item()
            scheduler.step()

            model.eval()
            val_rows = []
            with torch.no_grad():
                for batch in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{args.epochs} [Val]", leave=False):
                    images = batch["image"].to(device)
                    masks = batch["mask"].to(device)
                    seg_logits, enhanced, aux = forward_joint(model, images, args.joint_model)
                    val_rows.extend(
                        per_image_metrics_from_logits(
                            seg_logits,
                            masks,
                            threshold=args.threshold,
                            boundary_tolerance=args.boundary_tolerance,
                        )
                    )
            val_avg = average_metric_rows(val_rows)
            n_batches = max(len(train_loader), 1)
            weights = criterion.get_distill_weights()
            line = (
                f"Epoch {epoch + 1:03d} | LR={scheduler.get_last_lr()[0]:.6f} | "
                f"TrainLoss={train_loss / n_batches:.4f} | "
                f"Seg={train_seg / n_batches:.4f} | MSE={train_mse / n_batches:.4f} | "
                f"Grad={train_grad / n_batches:.4f} | Bnd={train_boundary / n_batches:.4f} | "
                f"W(MSE={weights['mse']:.2f},Grad={weights['grad']:.2f}) | "
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
    print("\n[SMP Joint Evaluation Complete]")
    print(f"Aggregate CSV: {aggregate_path}")
    print(
        f"{summary['experiment']}: Dice={summary['dice']:.4f}, IoU={summary['iou']:.4f}, "
        f"Recall={summary['sensitivity']:.4f}, Precision={summary['precision']:.4f}, "
        f"HD95={summary['hd95']:.2f}, clDice={summary['cldice']:.4f}, BoundaryF1={summary['boundary_f1']:.4f}"
    )


if __name__ == "__main__":
    main()

