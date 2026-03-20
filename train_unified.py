"""
统一训练脚本 - 支持 Baseline 和 Ours 模式
"""
import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# 确保导入项目的 utils，而不是 TransUNet 的
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datasets.dataset_vessel import VesselDataset
from utils.metrics import calculate_comprehensive_metrics
from models.transunet_official import TransUNetOfficial
from models.joint_framework import Enhancer, JointModel
from losses.joint_loss import JointDistillationLoss

def get_args():
    parser = argparse.ArgumentParser(description="统一训练脚本")

    # 模式选择
    parser.add_argument("--mode", type=str, default="baseline",
                        choices=["baseline", "ours"],
                        help="训练模式: baseline(纯TransUNet) 或 ours(联合蒸馏)")

    # 数据路径
    parser.add_argument("--data_dir", type=str, default="./dataset_raw_split")
    parser.add_argument("--save_dir", type=str, default="./results/experiments")

    # 训练参数
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)

    # 蒸馏权重（仅 ours 模式使用）
    parser.add_argument("--lambda_mse", type=float, default=10.0)
    parser.add_argument("--lambda_grad", type=float, default=30.0)

    # 预训练权重
    parser.add_argument("--pretrained", type=str,
                        default="model/vit_checkpoint/imagenet21k/R50+ViT-B_16.npz",
                        help="TransUNet预训练权重路径")
    parser.add_argument("--no_pretrained", action="store_true",
                        help="不使用预训练权重（从头训练）")

    return parser.parse_args()

def main():
    args = get_args()
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # 创建保存目录
    exp_name = f"{args.mode}_transunet"
    save_path = os.path.join(args.save_dir, exp_name)
    os.makedirs(save_path, exist_ok=True)

    print(f"[*] 训练模式: {args.mode.upper()}")
    print(f"[*] 保存路径: {save_path}")
    print(f"[*] 设备: {DEVICE}")

    # ================= 1. 数据加载 =================
    # Ours 模式需要 Teacher Prior
    teacher_dir = os.path.join(args.data_dir, "train/teacher_priors") if args.mode == "ours" else None

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
    pretrained_path = None if args.no_pretrained else args.pretrained

    segmentor = TransUNetOfficial(
        n_channels=3, n_classes=1, img_size=256,
        pretrained_path=pretrained_path
    )

    if args.mode == "baseline":
        # Baseline: 纯 TransUNet (BCE + Dice)
        model = segmentor.to(DEVICE)
        bce_loss = nn.BCEWithLogitsLoss()

        def criterion(outputs, masks):
            loss_bce = bce_loss(outputs, masks)
            pred_sig = torch.sigmoid(outputs)
            intersection = (pred_sig * masks).sum()
            dice_loss = 1 - (2. * intersection + 1e-6) / (pred_sig.sum() + masks.sum() + 1e-6)
            return loss_bce + dice_loss
    else:
        # Ours: Enhancer + TransUNet + 联合蒸馏
        enhancer = Enhancer(in_channels=3, out_channels=3)
        model = JointModel(enhancer, segmentor).to(DEVICE)
        criterion = JointDistillationLoss(
            lambda_mse=args.lambda_mse,
            lambda_grad=args.lambda_grad
        ).to(DEVICE)

    # ================= 3. 优化器 =================
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    # 日志文件
    log_file = open(os.path.join(save_path, "training_log.txt"), "w", encoding="utf-8")
    log_file.write(f"=== {args.mode.upper()} MODE ===\n")
    log_file.write(f"Epochs: {args.epochs}, Batch: {args.batch_size}, LR: {args.lr}\n")
    if args.mode == "ours":
        log_file.write(f"Lambda MSE: {args.lambda_mse}, Lambda Grad: {args.lambda_grad}\n")
    log_file.write("\n")

    best_dice = 0.0
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
            log_str = (f"Ep {epoch+1:03d} | Loss(Tot:{avg_loss:.3f} Seg:{train_seg/N:.3f} "
                      f"MSE:{train_mse/N:.3f} Grad:{train_grad/N:.3f}) | "
                      f"Dice: {avg_metrics['dice']:.4f} | HD95: {avg_metrics['hd95']:.2f}")
        print(log_str)
        log_file.write(log_str + "\n")
        log_file.flush()

        # 保存最优模型
        if avg_metrics['dice'] > best_dice:
            best_dice = avg_metrics['dice']
            torch.save(model.state_dict(), os.path.join(save_path, "best_model.pth"))
            print(f"[*] 更新最优模型: Dice = {best_dice:.4f}")

    log_file.close()

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



