"""
Unified training script for baseline TransUNet and green-prior joint models.
"""
import argparse
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
    OutputDistillationLoss,
    build_segmentation_loss,
)
from utils.metrics import average_metric_rows, per_image_metrics_from_logits
from models.joint_framework import (
    AnisotropicEnhancer,
    Enhancer,
    JointModel,
    JointModel_BoundaryRefine,
    JointModel_DecoderDistill,
    JointModel_DecoderDistillV2,
    JointModel_DualFusion,
    JointModel_Gated,
    JointModel_V2,
    MultiScaleEnhancer,
)
from models.transunet_official import TransUNetOfficial
from models.green_prior_fusion import GreenPriorFusionModel, PRIOR_FUSION_VARIANTS


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
    parser.add_argument(
        "--mode",
        type=str,
        default="baseline",
        choices=["baseline", "ours", "prior_fusion", "soft_kd"],
    )
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
    parser.add_argument("--joint_model", type=str, default="v1", choices=["v1", "v2", "gated", "boundary_refine", "decoder_distill", "decoder_distill_v2", "dual_fusion"])
    parser.add_argument("--attention_mode", type=str, default="normal", choices=["normal", "inverse"])
    parser.add_argument("--boundary_aux_weight", type=float, default=0.3)
    parser.add_argument("--lambda_decoder_distill", type=float, default=1.0)
    parser.add_argument("--decoder_distill_layers", type=str, default="2,3")
    parser.add_argument("--decoder_distill_mode", type=str, default="mse", choices=["mse", "normalized_mse", "cosine", "cosine_mse"])
    parser.add_argument("--decoder_teacher_weight", type=str, default="")
    parser.add_argument("--decoder_teacher_pretrained", type=str, default="")
    parser.add_argument("--intensity_aug", type=str, default="on", choices=["on", "off"])
    parser.add_argument("--pretrained", type=str, default="")
    parser.add_argument("--init_weight", type=str, default="")
    parser.add_argument("--soft_target_dir", type=str, default="")
    parser.add_argument("--disagreement_dir", type=str, default="")
    parser.add_argument("--lambda_kd", type=float, default=0.3)
    parser.add_argument("--kd_weight_mode", type=str, default="uniform", choices=["uniform", "agreement"])
    parser.add_argument(
        "--prior_fusion_variant",
        type=str,
        default="plain_single",
        choices=PRIOR_FUSION_VARIANTS,
    )
    parser.add_argument(
        "--evaluate_test_after_training",
        action="store_true",
        help="Evaluate test after training. Keep disabled during model selection to avoid test exposure.",
    )
    return parser.parse_args()


def get_teacher_dir(data_dir, teacher_mode):
    if teacher_mode == "green+clahe" and os.path.isdir(os.path.join(data_dir, "train", "teacher_priors")):
        return os.path.join(data_dir, "train", "teacher_priors")
    teacher_suffix = teacher_mode.replace("+", "_")
    return os.path.join(data_dir, "train", f"teacher_priors_{teacher_suffix}")


def forward_for_logits(model, images, mode, joint_model):
    if mode in ["baseline", "prior_fusion", "soft_kd"]:
        return model(images)
    if joint_model in ["v2", "gated", "boundary_refine", "decoder_distill", "decoder_distill_v2", "dual_fusion"]:
        outputs, _, _ = model(images)
    else:
        outputs, _ = model(images)
    return outputs


def load_torch_state_dict(model, weight_path, device, label):
    try:
        state_dict = torch.load(weight_path, map_location=device, weights_only=True)
    except TypeError:
        state_dict = torch.load(weight_path, map_location=device)
    model.load_state_dict(state_dict)
    print(f"[*] loaded {label}: {weight_path}")


def build_enhancer(args):
    if args.enhancer == "multiscale":
        return MultiScaleEnhancer(in_channels=3, out_channels=3, norm_type=args.enhancer_norm)
    if args.enhancer == "anisotropic":
        return AnisotropicEnhancer(in_channels=3, out_channels=3, norm_type=args.enhancer_norm)
    return Enhancer(in_channels=3, out_channels=3, norm_type=args.enhancer_norm)


def evaluate_per_image(model, loader, mode, joint_model):
    """Average metrics over images so the final partial batch has the correct weight."""
    rows = []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(next(model.parameters()).device)
            masks = batch["mask"].to(images.device)
            outputs = forward_for_logits(model, images, mode, joint_model)
            rows.extend(per_image_metrics_from_logits(outputs, masks))
    return average_metric_rows(rows)


def main():
    args = get_args()
    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not args.data_dir:
        args.data_dir = DATASETS[args.dataset]

    timestamp = datetime.now().strftime("%m%d_%H%M")
    if args.exp_name:
        exp_name = args.exp_name
    elif args.mode == "prior_fusion":
        exp_name = f"{args.dataset}/prior_fusion_{args.prior_fusion_variant}"
    elif args.mode == "soft_kd":
        exp_name = f"{args.dataset}/soft_kd_{args.kd_weight_mode}_lambda{args.lambda_kd:g}"
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
    if args.mode == "prior_fusion":
        print(f"[*] prior_fusion_variant: {args.prior_fusion_variant}")
    if args.mode == "soft_kd":
        print(f"[*] init_weight: {args.init_weight}")
        print(f"[*] soft_target_dir: {args.soft_target_dir}")
        print(f"[*] kd: mode={args.kd_weight_mode}, lambda={args.lambda_kd}")

    soft_target_dir = args.soft_target_dir or None
    disagreement_dir = args.disagreement_dir or None
    if args.mode == "soft_kd":
        if not args.init_weight:
            raise ValueError("--init_weight is required for soft_kd")
        if not os.path.isfile(args.init_weight):
            raise FileNotFoundError(f"Missing student initialization weight: {args.init_weight}")
        if soft_target_dir is None:
            raise ValueError("--soft_target_dir is required for soft_kd")
        if args.kd_weight_mode == "agreement" and disagreement_dir is None:
            raise ValueError("--disagreement_dir is required for agreement-weighted soft_kd")

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
        soft_target_dir=soft_target_dir if args.mode == "soft_kd" else None,
        disagreement_dir=disagreement_dir if args.mode == "soft_kd" else None,
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
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = None
    if args.evaluate_test_after_training:
        test_dataset = VesselDataset(
            image_dir=os.path.join(args.data_dir, "test/images"),
            mask_dir=os.path.join(args.data_dir, "test/masks"),
            teacher_dir=None,
            img_size=256,
            augment=False,
        )
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    pretrained_path = args.pretrained if args.pretrained else None
    segmentor = TransUNetOfficial(n_channels=3, n_classes=1, img_size=256, pretrained_path=pretrained_path)

    if args.mode in ["baseline", "prior_fusion", "soft_kd"]:
        if args.mode == "prior_fusion":
            model = GreenPriorFusionModel(
                segmentor,
                variant=args.prior_fusion_variant,
            ).to(device)
        else:
            model = segmentor.to(device)
        segmentation_criterion = build_segmentation_loss(
            seg_loss=args.seg_loss,
            cldice_weight=args.cldice_weight,
            boundary_weight=args.boundary_weight,
            cbdice_weight=args.cbdice_weight,
            focal_alpha=args.focal_alpha,
            focal_beta=args.focal_beta,
            focal_gamma=args.focal_gamma,
        ).to(device)
        if args.mode == "soft_kd":
            load_torch_state_dict(model, args.init_weight, device, "student initialization weight")
            criterion = OutputDistillationLoss(
                segmentation_criterion,
                lambda_kd=args.lambda_kd,
                weight_mode=args.kd_weight_mode,
            ).to(device)
        else:
            criterion = segmentation_criterion
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
        elif args.joint_model == "decoder_distill_v2":
            if not args.decoder_teacher_weight:
                raise ValueError("--decoder_teacher_weight is required for decoder_distill_v2")
            teacher_pretrained = args.decoder_teacher_pretrained or pretrained_path
            teacher_segmentor = TransUNetOfficial(
                n_channels=3,
                n_classes=1,
                img_size=256,
                pretrained_path=teacher_pretrained,
            )
            load_torch_state_dict(
                teacher_segmentor,
                args.decoder_teacher_weight,
                device,
                "decoder teacher weight",
            )
            model = JointModel_DecoderDistillV2(enhancer, segmentor, teacher_segmentor).to(device)
        elif args.joint_model == "dual_fusion":
            model = JointModel_DualFusion(enhancer, segmentor, norm_type=args.enhancer_norm).to(device)
        else:
            model = JointModel(enhancer, segmentor).to(device)
        if args.joint_model == "boundary_refine":
            criterion_class = JointDistillationBoundaryLoss
            criterion_kwargs = {"boundary_aux_weight": args.boundary_aux_weight}
        elif args.joint_model in ["decoder_distill", "decoder_distill_v2"]:
            criterion_class = JointDecoderDistillationLoss
            criterion_kwargs = {
                "lambda_decoder_distill": args.lambda_decoder_distill,
                "decoder_distill_layers": args.decoder_distill_layers,
                "decoder_distill_mode": args.decoder_distill_mode,
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

    optim_params = [param for param in model.parameters() if param.requires_grad]
    if args.mode == "ours" and args.loss_weighting == "learnable":
        optim_params += list(criterion.parameters())
    optimizer = optim.AdamW(optim_params, lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    config = vars(args).copy()
    config["device"] = device
    with open(os.path.join(save_path, "config.json"), "w", encoding="utf-8") as config_file:
        json.dump(config, config_file, ensure_ascii=False, indent=2)

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
            log_file.write(f"Decoder Distill Mode: {args.decoder_distill_mode}\n")
            log_file.write(f"Decoder Teacher Weight: {args.decoder_teacher_weight}\n")
            log_file.write(f"Intensity Aug: {args.intensity_aug}\n")
        elif args.mode == "prior_fusion":
            log_file.write(f"Prior Fusion Variant: {args.prior_fusion_variant}\n")
            log_file.write(f"Intensity Aug: {args.intensity_aug}\n")
        elif args.mode == "soft_kd":
            log_file.write(f"Student Init Weight: {args.init_weight}\n")
            log_file.write(f"Soft Target Dir: {args.soft_target_dir}\n")
            log_file.write(f"Disagreement Dir: {args.disagreement_dir}\n")
            log_file.write(f"KD Weight Mode: {args.kd_weight_mode}\n")
            log_file.write(f"Lambda KD: {args.lambda_kd}\n")
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
            train_kd = 0.0

            for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs} [Train]", leave=False):
                images = batch["image"].to(device)
                masks = batch["mask"].to(device)
                optimizer.zero_grad()

                if args.mode in ["baseline", "prior_fusion"]:
                    outputs = model(images)
                    loss = criterion(outputs, masks)
                elif args.mode == "soft_kd":
                    outputs = model(images)
                    soft_targets = batch["soft_target"].to(device)
                    disagreements = batch.get("disagreement")
                    if disagreements is not None:
                        disagreements = disagreements.to(device)
                    loss, l_seg, l_kd = criterion(outputs, masks, soft_targets, disagreements)
                    train_seg += l_seg.item()
                    train_kd += l_kd.item()
                else:
                    teachers = batch["teacher"].to(device)
                    if args.joint_model == "boundary_refine":
                        seg_preds, enhanced_imgs, boundary_logits = model(images)
                        loss, l_seg, l_mse, l_grad, l_boundary = criterion(seg_preds, masks, enhanced_imgs, teachers, boundary_logits)
                        train_boundary += l_boundary.item()
                    elif args.joint_model in ["decoder_distill", "decoder_distill_v2"]:
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
            avg_metrics = evaluate_per_image(model, val_loader, args.mode, args.joint_model)
            if args.mode in ["baseline", "prior_fusion"]:
                log_str = f"Ep {epoch + 1:03d} | Loss: {avg_loss:.4f} | Dice: {avg_metrics['dice']:.4f} | HD95: {avg_metrics['hd95']:.2f}"
                if args.mode == "prior_fusion":
                    diagnostics = model.fusion_diagnostics()
                    diagnostic_text = " ".join(
                        f"{name}:{value:.3f}" for name, value in sorted(diagnostics.items())
                    )
                    log_str += f" | Prior({diagnostic_text})"
            elif args.mode == "soft_kd":
                n_batches = len(train_loader)
                log_str = (
                    f"Ep {epoch + 1:03d} | Loss(Tot:{avg_loss:.3f} "
                    f"Seg:{train_seg / n_batches:.3f} KD:{train_kd / n_batches:.3f}) | "
                    f"W(KD:{args.lambda_kd:.2f}) | Dice: {avg_metrics['dice']:.4f} | "
                    f"HD95: {avg_metrics['hd95']:.2f}"
                )
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

    if args.evaluate_test_after_training:
        print("\n" + "=" * 50)
        print("[*] Load best model for requested test evaluation...")
        model.load_state_dict(torch.load(os.path.join(save_path, "best_model.pth"), map_location=device, weights_only=True))
        model.eval()
        avg_test_metrics = evaluate_per_image(model, test_loader, args.mode, args.joint_model)
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
