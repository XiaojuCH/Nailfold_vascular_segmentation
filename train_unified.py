"""
Unified training script for baseline TransUNet and green-prior joint models.
"""
import argparse
import os
import random
import sys
from datetime import datetime

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datasets.dataset_vessel import VesselDataset
from losses.joint_loss import (
    JointDecoderDistillationLoss,
    JointDistillationBoundaryLoss,
    JointDistillationLoss,
    build_segmentation_loss,
)
from utils.metrics import calculate_comprehensive_metrics
from models.joint_framework import (
    AnisotropicEnhancer,
    Enhancer,
    JointModel,
    JointModel_BoundaryRefine,
    JointModel_DecoderDistill,
    JointModel_DualFusion,
    JointModel_Gated,
    JointModel_V2,
    MultiScaleEnhancer,
)
from models.transunet_official import TransUNetOfficial


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


def get_args():
    parser = argparse.ArgumentParser(description="Unified training script")
    parser.add_argument("--mode", type=str, default="baseline", choices=["baseline", "ours"])
    parser.add_argument("--exp_name", type=str, default="")

    parser.add_argument("--dataset", type=str, default="jiabi", choices=list(DATASETS.keys()))
    parser.add_argument("--data_dir", type=str, default="")
    parser.add_argument("--save_dir", type=str, default="./results/experiments")

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--seg_loss", type=str, default="bce_dice", choices=SEG_LOSSES)
    parser.add_argument("--cldice_weight", type=float, default=0.5)
    parser.add_argument("--boundary_weight", type=float, default=0.5)
    parser.add_argument("--cbdice_weight", type=float, default=0.5)
    parser.add_argument("--focal_alpha", type=float, default=0.3)
    parser.add_argument("--focal_beta", type=float, default=0.7)
    parser.add_argument("--focal_gamma", type=float, default=0.75)

    parser.add_argument("--lambda_mse", type=float, default=10.0)
    parser.add_argument("--lambda_grad", type=float, default=30.0)
    parser.add_argument("--loss_weighting", type=str, default="fixed", choices=["fixed", "learnable"])

    parser.add_argument("--teacher_mode", type=str, default="green+clahe", choices=TEACHER_MODES)
    parser.add_argument("--enhancer", type=str, default="basic", choices=["basic", "multiscale", "anisotropic"])
    parser.add_argument("--enhancer_norm", type=str, default="bn", choices=["bn", "none"])
    parser.add_argument("--joint_model", type=str, default="v1", choices=["v1", "v2", "gated", "boundary_refine", "decoder_distill", "dual_fusion"])
    parser.add_argument("--attention_mode", type=str, default="normal", choices=["normal", "inverse"])
    parser.add_argument("--boundary_aux_weight", type=float, default=0.3)
    parser.add_argument("--lambda_decoder_distill", type=float, default=1.0)
    parser.add_argument("--decoder_distill_layers", type=str, default="2,3")
    parser.add_argument("--intensity_aug", type=str, default="on", choices=["on", "off"])
    parser.add_argument("--pretrained", type=str, default="")
    return parser.parse_args()


def get_teacher_dir(data_dir, teacher_mode):
    if teacher_mode == "green+clahe" and os.path.isdir(os.path.join(data_dir, "train", "teacher_priors")):
        return os.path.join(data_dir, "train", "teacher_priors")
    teacher_suffix = teacher_mode.replace("+", "_")
    return os.path.join(data_dir, "train", f"teacher_priors_{teacher_suffix}")


def forward_for_logits(model, images, mode, joint_model):
    if mode == "baseline":
        return model(images)
    if joint_model in ["v2", "gated", "boundary_refine", "decoder_distill", "dual_fusion"]:
        outputs, _, _ = model(images)
    else:
        outputs, _ = model(images)
    return outputs


def build_enhancer(args):
    if args.enhancer == "multiscale":
        return MultiScaleEnhancer(in_channels=3, out_channels=3, norm_type=args.enhancer_norm)
    if args.enhancer == "anisotropic":
        return AnisotropicEnhancer(in_channels=3, out_channels=3, norm_type=args.enhancer_norm)
    return Enhancer(in_channels=3, out_channels=3, norm_type=args.enhancer_norm)


def main():
    args = get_args()
    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not args.data_dir:
        args.data_dir = DATASETS[args.dataset]

    timestamp = datetime.now().strftime("%m%d_%H%M")
    if args.exp_name:
        exp_name = args.exp_name
    else:
        suffix = args.teacher_mode.replace("+", "_")
        exp_name = f"{args.dataset}/{args.mode}" if args.teacher_mode == "green+clahe" else f"{args.dataset}/{args.mode}_{suffix}"
    save_path = os.path.join(args.save_dir, exp_name, timestamp)
    os.makedirs(save_path, exist_ok=True)

    print(f"[*] mode: {args.mode.upper()}")
    print(f"[*] dataset: {args.dataset} ({args.data_dir})")
    print(f"[*] save_path: {save_path}")
    print(f"[*] device: {device}")
    print(f"[*] seed: {args.seed}")
    print(f"[*] seg_loss: {args.seg_loss}")
    if args.pretrained:
        print(f"[*] pretrained: {args.pretrained}")

    teacher_dir = get_teacher_dir(args.data_dir, args.teacher_mode) if args.mode == "ours" else None
    if args.mode == "ours":
        print(f"[*] teacher_mode: {args.teacher_mode}")
        print(f"[*] teacher_dir: {teacher_dir}")
        if not os.path.isdir(teacher_dir):
            raise FileNotFoundError(f"Missing teacher prior directory: {teacher_dir}")

    train_dataset = VesselDataset(
        image_dir=os.path.join(args.data_dir, "train/images"),
        mask_dir=os.path.join(args.data_dir, "train/masks"),
        teacher_dir=teacher_dir,
        img_size=256,
        augment=True,
        intensity_aug=args.intensity_aug == "on",
    )
    val_dataset = VesselDataset(
        image_dir=os.path.join(args.data_dir, "val/images"),
        mask_dir=os.path.join(args.data_dir, "val/masks"),
        teacher_dir=None,
        img_size=256,
        augment=False,
    )
    test_dataset = VesselDataset(
        image_dir=os.path.join(args.data_dir, "test/images"),
        mask_dir=os.path.join(args.data_dir, "test/masks"),
        teacher_dir=None,
        img_size=256,
        augment=False,
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    pretrained_path = args.pretrained if args.pretrained else None
    segmentor = TransUNetOfficial(n_channels=3, n_classes=1, img_size=256, pretrained_path=pretrained_path)

    if args.mode == "baseline":
        model = segmentor.to(device)
        criterion = build_segmentation_loss(
            seg_loss=args.seg_loss,
            cldice_weight=args.cldice_weight,
            boundary_weight=args.boundary_weight,
            cbdice_weight=args.cbdice_weight,
            focal_alpha=args.focal_alpha,
            focal_beta=args.focal_beta,
            focal_gamma=args.focal_gamma,
        ).to(device)
    else:
        enhancer = build_enhancer(args)
        if args.joint_model == "v2":
            model = JointModel_V2(enhancer, segmentor, attention_mode=args.attention_mode).to(device)
        elif args.joint_model == "gated":
            model = JointModel_Gated(enhancer, segmentor).to(device)
        elif args.joint_model == "boundary_refine":
            model = JointModel_BoundaryRefine(enhancer, segmentor).to(device)
        elif args.joint_model == "decoder_distill":
            model = JointModel_DecoderDistill(enhancer, segmentor).to(device)
        elif args.joint_model == "dual_fusion":
            model = JointModel_DualFusion(enhancer, segmentor, norm_type=args.enhancer_norm).to(device)
        else:
            model = JointModel(enhancer, segmentor).to(device)
        if args.joint_model == "boundary_refine":
            criterion_class = JointDistillationBoundaryLoss
            criterion_kwargs = {"boundary_aux_weight": args.boundary_aux_weight}
        elif args.joint_model == "decoder_distill":
            criterion_class = JointDecoderDistillationLoss
            criterion_kwargs = {
                "lambda_decoder_distill": args.lambda_decoder_distill,
                "decoder_distill_layers": args.decoder_distill_layers,
            }
        else:
            criterion_class = JointDistillationLoss
            criterion_kwargs = {}
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
    if args.mode == "ours" and args.loss_weighting == "learnable":
        optim_params += list(criterion.parameters())
    optimizer = optim.AdamW(optim_params, lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    log_path = os.path.join(save_path, "training_log.txt")
    with open(log_path, "w", encoding="utf-8") as log_file:
        log_file.write(f"=== {exp_name} ===\n")
        log_file.write(f"Mode: {args.mode.upper()}, Epochs: {args.epochs}, Batch: {args.batch_size}, LR: {args.lr}\n")
        log_file.write(f"Data: {args.data_dir}\n")
        log_file.write(f"Seed: {args.seed}\n")
        log_file.write(
            f"Seg Loss: {args.seg_loss}, clDice_w: {args.cldice_weight}, boundary_w: {args.boundary_weight}, "
            f"cbDice_w: {args.cbdice_weight}, focal: ({args.focal_alpha}, {args.focal_beta}, {args.focal_gamma})\n"
        )
        if args.mode == "ours":
            log_file.write(f"Teacher Mode: {args.teacher_mode}\n")
            log_file.write(f"Boundary Aux Weight: {args.boundary_aux_weight}\n")
            log_file.write(f"Lambda MSE: {args.lambda_mse}, Lambda Grad: {args.lambda_grad}\n")
            log_file.write(f"Loss Weighting: {args.loss_weighting}\n")
            log_file.write(f"Enhancer: {args.enhancer}\n")
            log_file.write(f"Enhancer Norm: {args.enhancer_norm}\n")
            log_file.write(f"Joint Model: {args.joint_model}\n")
            log_file.write(f"Attention Mode: {args.attention_mode}\n")
            log_file.write(f"Lambda Decoder Distill: {args.lambda_decoder_distill}\n")
            log_file.write(f"Decoder Distill Layers: {args.decoder_distill_layers}\n")
            log_file.write(f"Intensity Aug: {args.intensity_aug}\n")
        if args.pretrained:
            log_file.write(f"Pretrained: {args.pretrained}\n")
        log_file.write("\n")

        best_dice = -1.0
        patience_counter = 0

        for epoch in range(args.epochs):
            model.train()
            train_loss = 0.0
            train_seg = 0.0
            train_mse = 0.0
            train_grad = 0.0
            train_boundary = 0.0
            train_decoder = 0.0

            for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs} [Train]", leave=False):
                images = batch["image"].to(device)
                masks = batch["mask"].to(device)
                optimizer.zero_grad()

                if args.mode == "baseline":
                    outputs = model(images)
                    loss = criterion(outputs, masks)
                else:
                    teachers = batch["teacher"].to(device)
                    if args.joint_model == "boundary_refine":
                        seg_preds, enhanced_imgs, boundary_logits = model(images)
                        loss, l_seg, l_mse, l_grad, l_boundary = criterion(seg_preds, masks, enhanced_imgs, teachers, boundary_logits)
                        train_boundary += l_boundary.item()
                    elif args.joint_model == "decoder_distill":
                        seg_preds, enhanced_imgs, decoder_feature_pair = model(images, teachers)
                        loss, l_seg, l_mse, l_grad, l_decoder = criterion(seg_preds, masks, enhanced_imgs, teachers, decoder_feature_pair)
                        train_decoder += l_decoder.item()
                    elif args.joint_model in ["v2", "gated"]:
                        seg_preds, enhanced_imgs, _ = model(images)
                        loss, l_seg, l_mse, l_grad = criterion(seg_preds, masks, enhanced_imgs, teachers)
                    elif args.joint_model == "dual_fusion":
                        seg_preds, enhanced_imgs, _ = model(images)
                        loss, l_seg, l_mse, l_grad = criterion(seg_preds, masks, enhanced_imgs, teachers)
                    else:
                        seg_preds, enhanced_imgs = model(images)
                        loss, l_seg, l_mse, l_grad = criterion(seg_preds, masks, enhanced_imgs, teachers)
                    train_seg += l_seg.item()
                    train_mse += l_mse.item()
                    train_grad += l_grad.item()

                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            avg_loss = train_loss / len(train_loader)
            scheduler.step()

            model.eval()
            metrics_sum = {"dice": 0.0, "iou": 0.0, "hd95": 0.0}
            with torch.no_grad():
                for batch in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{args.epochs} [Val]", leave=False):
                    images = batch["image"].to(device)
                    masks = batch["mask"].to(device)
                    outputs = forward_for_logits(model, images, args.mode, args.joint_model)
                    batch_metrics = calculate_comprehensive_metrics(outputs, masks)
                    for key in metrics_sum:
                        metrics_sum[key] += batch_metrics[key]

            avg_metrics = {key: value / len(val_loader) for key, value in metrics_sum.items()}
            if args.mode == "baseline":
                log_str = f"Ep {epoch + 1:03d} | Loss: {avg_loss:.4f} | Dice: {avg_metrics['dice']:.4f} | HD95: {avg_metrics['hd95']:.2f}"
            else:
                n_batches = len(train_loader)
                weights = criterion.get_distill_weights()
                log_str = (
                    f"Ep {epoch + 1:03d} | Loss(Tot:{avg_loss:.3f} Seg:{train_seg / n_batches:.3f} "
                    f"MSE:{train_mse / n_batches:.3f} Grad:{train_grad / n_batches:.3f} "
                    f"Bnd:{train_boundary / n_batches:.3f} Dec:{train_decoder / n_batches:.3f}) | "
                    f"W(MSE:{weights['mse']:.2f} Grad:{weights['grad']:.2f}) | "
                    f"Dice: {avg_metrics['dice']:.4f} | HD95: {avg_metrics['hd95']:.2f}"
                )
            print(log_str)
            log_file.write(log_str + "\n")
            log_file.flush()

            if avg_metrics["dice"] > best_dice:
                best_dice = avg_metrics["dice"]
                patience_counter = 0
                torch.save(model.state_dict(), os.path.join(save_path, "best_model.pth"))
                print(f"[*] Update best model: Dice = {best_dice:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print(f"[*] Early stopping after {args.patience} epochs without improvement")
                    break

        if args.mode == "ours":
            final_weights = criterion.get_distill_weights()
            final_weight_str = f"[FINAL DISTILL WEIGHTS] MSE: {final_weights['mse']:.4f}, Grad: {final_weights['grad']:.4f}"
            print(final_weight_str)
            log_file.write(final_weight_str + "\n")

    print("\n" + "=" * 50)
    print("[*] Load best model for test evaluation...")
    model.load_state_dict(torch.load(os.path.join(save_path, "best_model.pth"), map_location=device, weights_only=True))
    model.eval()

    test_metrics_sum = {
        "dice": 0.0,
        "iou": 0.0,
        "accuracy": 0.0,
        "precision": 0.0,
        "sensitivity": 0.0,
        "specificity": 0.0,
        "hd95": 0.0,
    }
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="[*] Test"):
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            outputs = forward_for_logits(model, images, args.mode, args.joint_model)
            batch_metrics = calculate_comprehensive_metrics(outputs, masks)
            for key in test_metrics_sum:
                test_metrics_sum[key] += batch_metrics[key]

    avg_test_metrics = {key: value / len(test_loader) for key, value in test_metrics_sum.items()}
    test_result_str = (
        f"\n[FINAL TEST RESULTS] {args.mode.upper()}\n"
        f"--------------------------------------------------\n"
        f"Dice:  {avg_test_metrics['dice']:.4f}\n"
        f"IoU:   {avg_test_metrics['iou']:.4f}\n"
        f"HD95:  {avg_test_metrics['hd95']:.2f}\n"
        f"Sens:  {avg_test_metrics['sensitivity']:.4f}\n"
        f"Spec:  {avg_test_metrics['specificity']:.4f}\n"
        f"Acc:   {avg_test_metrics['accuracy']:.4f}\n"
        f"Prec:  {avg_test_metrics['precision']:.4f}\n"
        f"--------------------------------------------------"
    )
    print(test_result_str)
    with open(log_path, "a", encoding="utf-8") as log_file:
        log_file.write(test_result_str + "\n")

    print(f"\n[*] Training complete. Results saved to: {save_path}")


if __name__ == "__main__":
    main()
