"""
统一训练脚本 - 支持 Baseline 和 Ours 模式

References:
- TransUNet: Chen et al. "TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation"
  Medical Image Analysis, 2021. https://github.com/Beckschen/TransUNet
"""
import os
import sys
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

def set_seed(seed=42):
    """设置随机种子以确保可复现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 确保导入项目的 utils，而不是 TransUNet 的
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datasets.dataset_vessel import VesselDataset
from utils.metrics import calculate_comprehensive_metrics
from models.transunet_official import TransUNetOfficial
from models.joint_framework import Enhancer, MultiScaleEnhancer, JointModel, JointModel_V2, JointModel_Gated
from losses.joint_loss import JointDistillationLoss, build_segmentation_loss

def get_args():
    parser = argparse.ArgumentParser(description="统一训练脚本")

    # 模式选择
    parser.add_argument("--mode", type=str, default="baseline",
                        choices=["baseline", "ours"],
                        help="训练模式: baseline(纯TransUNet) 或 ours(联合蒸馏)")

    # 实验名称（用于区分不同对比实验/消融实验）
    parser.add_argument("--exp_name", type=str, default="",
                        help="实验名称，留空则自动生成 {mode}_transunet")

    # 数据集选择
    DATASETS = {
        "jiabi":         "./dataset_raw_split",
        "anfc256":       "./dataset_anfc256_split",
        "all":           "./dataset_all_split",
        "all_filtered":  "./dataset_all_filtered",
        "all_filtered_VT_Turn":  "./dataset_all_filtered_VT_Turn",
    }
    parser.add_argument("--dataset", type=str, default="jiabi",
                        choices=list(DATASETS.keys()),
                        help="数据集: jiabi(JiaBi) / anfc256(纯ANFC,68患者) / all(混合数据集) / all_filtered(连通域筛选后)")
    parser.add_argument("--data_dir", type=str, default="",
                        help="自定义数据集路径，留空则使用 --dataset 对应的默认路径")
    parser.add_argument("--save_dir", type=str, default="./results/experiments")

    # 训练参数
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=20, help="早停耐心值")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42, help="随机种子")

    # 分割损失，可用于结构/边界消融
    parser.add_argument("--seg_loss", type=str, default="bce_dice",
                        choices=["bce_dice", "focal_tversky", "unified_focal",
                                 "bce_dice_cldice", "bce_dice_boundary", "bce_dice_cldice_boundary"],
                        help="分割损失类型")
    parser.add_argument("--cldice_weight", type=float, default=0.5, help="soft-clDice loss 权重")
    parser.add_argument("--boundary_weight", type=float, default=0.5, help="soft boundary loss 权重")
    parser.add_argument("--focal_alpha", type=float, default=0.3, help="Focal Tversky alpha/FP 权重")
    parser.add_argument("--focal_beta", type=float, default=0.7, help="Focal Tversky beta/FN 权重")
    parser.add_argument("--focal_gamma", type=float, default=0.75, help="Focal/Tversky gamma")

    # 蒸馏权重（仅 ours 模式使用）
    parser.add_argument("--lambda_mse", type=float, default=10.0)
    parser.add_argument("--lambda_grad", type=float, default=30.0)
    parser.add_argument("--loss_weighting", type=str, default="fixed",
                        choices=["fixed", "learnable"],
                        help="ours 模式蒸馏损失权重: fixed(固定lambda) / learnable(可学习权重)")

    parser.add_argument("--teacher_mode", type=str, default="green+clahe",
                        choices=["green+clahe", "clahe_only", "green_only"],
                        help="消融实验：教师先验生成方式")
    parser.add_argument("--enhancer", type=str, default="basic",
                        choices=["basic", "multiscale"],
                        help="ours 模式增强器结构: basic(原始轻量Enhancer) / multiscale(多尺度分支Enhancer)")
    parser.add_argument("--joint_model", type=str, default="v1",
                        choices=["v1", "v2", "gated"],
                        help="ours 模式联合框架: v1(增强图直接分割) / v2(空间注意力残差融合) / gated(原图与增强图自适应门控融合)")
    parser.add_argument("--attention_mode", type=str, default="normal",
                        choices=["normal", "inverse"],
                        help="仅 joint_model=v2 时生效: normal(原始注意力) / inverse(反向注意力)")
    parser.add_argument("--pretrained", type=str, default="",
                        help="TransUNet预训练权重路径，留空则从头训练（默认）。"
                             "例: model/vit_checkpoint/imagenet21k/R50+ViT-B_16.npz")

    return parser.parse_args()

def main():
    args = get_args()

    # 设置随机种子以确保可复现性
    set_seed(args.seed)

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # 数据集路径处理
    DATASETS = {
        "jiabi":         "./dataset_raw_split",
        "anfc256":       "./dataset_anfc256_split",
        "all":           "./dataset_all_split",
        "all_filtered":  "./dataset_all_filtered",
        "all_filtered_VT_Turn":  "./dataset_all_filtered_VT_Turn",
    }
    if not args.data_dir:
        args.data_dir = DATASETS[args.dataset]

    from datetime import datetime
    timestamp = datetime.now().strftime("%m%d_%H%M")

    # 创建保存目录（按数据集和方法分类，加时间戳避免覆盖）
    if args.exp_name:
        exp_name = args.exp_name
    else:
        exp_name = f"{args.dataset}/{args.mode}" if args.teacher_mode == "green+clahe" else f"{args.dataset}/{args.mode}_{args.teacher_mode.replace('+', '_')}"
    save_path = os.path.join(args.save_dir, exp_name, timestamp)
    os.makedirs(save_path, exist_ok=True)

    print(f"[*] 训练模式: {args.mode.upper()}")
    print(f"[*] 数据集: {args.dataset} ({args.data_dir})")
    print(f"[*] 保存路径: {save_path}")
    print(f"[*] 设备: {DEVICE}")
    print(f"[*] 随机种子: {args.seed}")
    print(f"[*] 分割损失: {args.seg_loss}")

    # ================= 1. 数据加载 =================
    # Ours 模式需要 Teacher Prior，根据 teacher_mode 选择对应目录
    teacher_suffix = args.teacher_mode.replace("+", "_")
    teacher_priors_dir = f"teacher_priors_{teacher_suffix}" if args.teacher_mode != "green+clahe" else "teacher_priors"
    teacher_dir = os.path.join(args.data_dir, f"train/{teacher_priors_dir}") if args.mode == "ours" else None

    train_dataset = VesselDataset(
        image_dir=os.path.join(args.data_dir, "train/images"),
        mask_dir=os.path.join(args.data_dir, "train/masks"),
        teacher_dir=teacher_dir,
        img_size=256, augment=True
    )

    val_dataset = VesselDataset(
        image_dir=os.path.join(args.data_dir, "val/images"),
        mask_dir=os.path.join(args.data_dir, "val/masks"),
        teacher_dir=None, img_size=256, augment=False
    )

    test_dataset = VesselDataset(
        image_dir=os.path.join(args.data_dir, "test/images"),
        mask_dir=os.path.join(args.data_dir, "test/masks"),
        teacher_dir=None, img_size=256, augment=False
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # ================= 2. 模型创建 =================
    pretrained_path = args.pretrained if args.pretrained else None

    segmentor = TransUNetOfficial(
        n_channels=3, n_classes=1, img_size=256,
        pretrained_path=pretrained_path
    )

    if args.mode == "baseline":
        # Baseline: 纯 TransUNet，可切换分割损失
        model = segmentor.to(DEVICE)
        criterion = build_segmentation_loss(
            seg_loss=args.seg_loss,
            cldice_weight=args.cldice_weight,
            boundary_weight=args.boundary_weight,
            focal_alpha=args.focal_alpha,
            focal_beta=args.focal_beta,
            focal_gamma=args.focal_gamma,
        ).to(DEVICE)
    else:
        # Ours: Enhancer + TransUNet + 联合蒸馏
        if args.enhancer == "multiscale":
            enhancer = MultiScaleEnhancer(in_channels=3, out_channels=3)
        else:
            enhancer = Enhancer(in_channels=3, out_channels=3)
        if args.joint_model == "v2":
            model = JointModel_V2(enhancer, segmentor, attention_mode=args.attention_mode).to(DEVICE)
        elif args.joint_model == "gated":
            model = JointModel_Gated(enhancer, segmentor).to(DEVICE)
        else:
            model = JointModel(enhancer, segmentor).to(DEVICE)
        criterion = JointDistillationLoss(
            lambda_mse=args.lambda_mse,
            lambda_grad=args.lambda_grad,
            weight_mode=args.loss_weighting,
            seg_loss=args.seg_loss,
            cldice_weight=args.cldice_weight,
            boundary_weight=args.boundary_weight,
            focal_alpha=args.focal_alpha,
            focal_beta=args.focal_beta,
            focal_gamma=args.focal_gamma,
        ).to(DEVICE)

    # ================= 3. 优化器 =================
    optim_params = list(model.parameters())
    if args.mode == "ours" and args.loss_weighting == "learnable":
        optim_params += list(criterion.parameters())
    optimizer = optim.AdamW(optim_params, lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    # 日志文件
    log_file = open(os.path.join(save_path, "training_log.txt"), "w", encoding="utf-8")
    log_file.write(f"=== {exp_name} ===\n")
    log_file.write(f"Mode: {args.mode.upper()}, Epochs: {args.epochs}, Batch: {args.batch_size}, LR: {args.lr}\n")
    log_file.write(f"Data: {args.data_dir}\n")
    log_file.write(f"Seed: {args.seed}\n")
    log_file.write(f"Seg Loss: {args.seg_loss}, clDice_w: {args.cldice_weight}, boundary_w: {args.boundary_weight}, focal: ({args.focal_alpha}, {args.focal_beta}, {args.focal_gamma})\n")
    if args.mode == "ours":
        log_file.write(f"Lambda MSE: {args.lambda_mse}, Lambda Grad: {args.lambda_grad}\n")
        log_file.write(f"Loss Weighting: {args.loss_weighting}\n")
        log_file.write(f"Enhancer: {args.enhancer}\n")
        log_file.write(f"Joint Model: {args.joint_model}\n")
        log_file.write(f"Attention Mode: {args.attention_mode}\n")
    log_file.write("\n")

    best_dice = -1.0
    patience_counter = 0
    history = {"train_loss": [], "val_dice": [], "val_hd95": []}

    # ================= 4. 训练循环 =================
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        train_seg, train_mse, train_grad = 0.0, 0.0, 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]", leave=False):
            images = batch["image"].to(DEVICE)
            masks = batch["mask"].to(DEVICE)

            optimizer.zero_grad()

            if args.mode == "baseline":
                # Baseline: 直接分割
                outputs = model(images)
                loss = criterion(outputs, masks)
            else:
                # Ours: 联合蒸馏
                teachers = batch["teacher"].to(DEVICE)
                if args.joint_model in ["v2", "gated"]:
                    seg_preds, enhanced_imgs, aux_map = model(images)
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

        # ================= 5. 验证 =================
        model.eval()
        metrics_sum = {"dice": 0, "iou": 0, "hd95": 0}

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]", leave=False):
                images = batch["image"].to(DEVICE)
                masks = batch["mask"].to(DEVICE)

                if args.mode == "baseline":
                    outputs = model(images)
                else:
                    if args.joint_model in ["v2", "gated"]:
                        outputs, _, _ = model(images)
                    else:
                        outputs, _ = model(images)

                batch_metrics = calculate_comprehensive_metrics(outputs, masks)
                for k in metrics_sum.keys():
                    metrics_sum[k] += batch_metrics[k]

        avg_metrics = {k: v / len(val_loader) for k, v in metrics_sum.items()}
        history["train_loss"].append(avg_loss)
        history["val_dice"].append(avg_metrics['dice'])
        history["val_hd95"].append(avg_metrics['hd95'])

        # 日志
        if args.mode == "baseline":
            log_str = f"Ep {epoch+1:03d} | Loss: {avg_loss:.4f} | Dice: {avg_metrics['dice']:.4f} | HD95: {avg_metrics['hd95']:.2f}"
        else:
            N = len(train_loader)
            weights = criterion.get_distill_weights()
            log_str = (f"Ep {epoch+1:03d} | Loss(Tot:{avg_loss:.3f} Seg:{train_seg/N:.3f} "
                      f"MSE:{train_mse/N:.3f} Grad:{train_grad/N:.3f}) | "
                      f"W(MSE:{weights['mse']:.2f} Grad:{weights['grad']:.2f}) | "
                      f"Dice: {avg_metrics['dice']:.4f} | HD95: {avg_metrics['hd95']:.2f}")
        print(log_str)
        log_file.write(log_str + "\n")
        log_file.flush()

        # 保存最优模型
        if avg_metrics['dice'] > best_dice:
            best_dice = avg_metrics['dice']
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(save_path, "best_model.pth"))
            print(f"[*] 更新最优模型: Dice = {best_dice:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"[*] 早停触发，连续 {args.patience} 个 epoch 无提升")
                break

    log_file.close()

    if args.mode == "ours":
        final_weights = criterion.get_distill_weights()
        final_weight_str = f"[FINAL DISTILL WEIGHTS] MSE: {final_weights['mse']:.4f}, Grad: {final_weights['grad']:.4f}"
        print(final_weight_str)
        with open(os.path.join(save_path, "training_log.txt"), "a", encoding="utf-8") as f:
            f.write(final_weight_str + "\n")

    # ================= 6. 测试集评估 =================
    print("\n" + "="*50)
    print("[*] 加载最佳模型进行测试...")
    model.load_state_dict(torch.load(os.path.join(save_path, "best_model.pth"), map_location=DEVICE, weights_only=True))
    model.eval()

    test_metrics_sum = {"dice": 0, "iou": 0, "accuracy": 0, "precision": 0, "sensitivity": 0, "specificity": 0, "hd95": 0}

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="[*] 测试集评估"):
            images = batch["image"].to(DEVICE)
            masks = batch["mask"].to(DEVICE)

            if args.mode == "baseline":
                outputs = model(images)
            else:
                if args.joint_model in ["v2", "gated"]:
                    outputs, _, _ = model(images)
                else:
                    outputs, _ = model(images)

            batch_metrics = calculate_comprehensive_metrics(outputs, masks)
            for k in test_metrics_sum.keys():
                test_metrics_sum[k] += batch_metrics[k]

    avg_test_metrics = {k: v / len(test_loader) for k, v in test_metrics_sum.items()}

    # 打印测试结果
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

    with open(os.path.join(save_path, "training_log.txt"), "a", encoding="utf-8") as f:
        f.write(test_result_str + "\n")

    print(f"\n[*] 训练完成！结果保存至: {save_path}")

if __name__ == "__main__":
    main()


