"""
Baseline模型对比训练脚本 - UNet 和 UNet++

References:
- UNet: Ronneberger et al. "U-Net: Convolutional Networks for Biomedical Image Segmentation"
  MICCAI, 2015. https://arxiv.org/abs/1505.04597
- UNet++: Zhou et al. "UNet++: A Nested U-Net Architecture for Medical Image Segmentation"
  DLMIA, 2018. https://arxiv.org/abs/1807.10165
"""
import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

def set_seed(seed=42):
    """设置随机种子以确保可复现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 导入你的 Dataset
from datasets.dataset_vessel import VesselDataset
from utils.metrics import calculate_comprehensive_metrics

# 导入候选网络 (确保它们都是 512 通道瓶颈层，保证参数量公平)
from models.unet_baseline import UNet
from models.unet_plus_plus import UNetPlusPlus
from models.transunet import TransUNet

def get_args():
    parser = argparse.ArgumentParser(description="Baseline Model Sweeper for Vessel Segmentation")
    parser.add_argument("--model", type=str, default="unet", choices=["unet", "unet++"],
                        help="选择要测试的基线网络（TransUNet请使用train_unified.py）")

    # 数据集选择
    DATASETS = {
        "jiabi":         "./dataset_raw_split",
        "anfc256":       "./dataset_anfc256_split",
        "all":           "./dataset_all_split",
        "all_filtered":  "./dataset_all_filtered",
    }
    parser.add_argument("--dataset", type=str, default="jiabi", choices=list(DATASETS.keys()),
                        help="数据集: jiabi / anfc256 / all / all_filtered(连通域筛选后)")
    parser.add_argument("--data_dir", type=str, default="", help="自定义数据集路径，留空则使用--dataset对应路径")
    parser.add_argument("--save_dir", type=str, default="./results/experiments", help="保存根目录")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)

    args = parser.parse_args()
    # 处理数据集路径
    if not args.data_dir:
        args.data_dir = DATASETS[args.dataset]
    return args

def main():
    args = get_args()

    # 设置随机种子以确保可复现性
    set_seed(42)

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 动态创建保存路径（按数据集/模型分类）
    model_save_dir = os.path.join(args.save_dir, args.dataset, args.model)
    os.makedirs(model_save_dir, exist_ok=True)

    # 强制加上 utf-8 编码，防止 Windows 报错
    log_file = open(os.path.join(model_save_dir, "training_log.txt"), "w", encoding="utf-8")

    print(f"[*] 启动 Baseline 摸底测试...")
    print(f"[*] 数据集: {args.dataset} ({args.data_dir})")
    print(f"[*] 当前测试网络: {args.model.upper()} | 设备: {DEVICE}")
    print(f"[*] 保存路径: {model_save_dir}")

    # ================= 1. 数据加载 =================
    train_dataset = VesselDataset(
        image_dir=os.path.join(args.data_dir, "train/images"),
        mask_dir=os.path.join(args.data_dir, "train/masks"),
        teacher_dir=None, 
        img_size=256,
        augment=True
    )
    val_dataset = VesselDataset(
        image_dir=os.path.join(args.data_dir, "val/images"),
        mask_dir=os.path.join(args.data_dir, "val/masks"),
        teacher_dir=None,
        img_size=256,
        augment=False
    )
    test_dataset = VesselDataset(
        image_dir=os.path.join(args.data_dir, "test/images"),
        mask_dir=os.path.join(args.data_dir, "test/masks"),
        teacher_dir=None,
        img_size=256,
        augment=False
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # ================= 2. 模型与优化器配置 =================
    if args.model == "unet":
        model = UNet(n_channels=3, n_classes=1).to(DEVICE)
    elif args.model == "unet++":
        model = UNetPlusPlus(n_channels=3, n_classes=1).to(DEVICE)
    elif args.model == "transunet":
        model = TransUNet(n_channels=3, n_classes=1, img_size=256).to(DEVICE)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    # ================= 写入超参数头部日志 =================
    hyperparams_str = (
        f"========================================\n"
        f"        TRAINING HYPERPARAMETERS        \n"
        f"========================================\n"
        f"Model:          {args.model.upper()}\n"
        f"Epochs:         {args.epochs}\n"
        f"Batch Size:     {args.batch_size}\n"
        f"Initial LR:     {args.lr}\n"
        f"Optimizer:      AdamW (weight_decay=1e-4)\n"
        f"Scheduler:      CosineAnnealingLR\n"
        f"Loss Function:  BCEWithLogitsLoss\n"
        f"Image Size:     256x256\n"
        f"========================================\n\n"
    )
    log_file.write(hyperparams_str)
    log_file.flush()

    best_dice = 0.0
    
    # 记录每个 Epoch 的数据用于画图
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_dice": [],
        "val_iou": []
    }

    # ================= 3. 训练循环 =================
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]", leave=False):
            images, masks = batch["image"].to(DEVICE), batch["mask"].to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # ================= 4. 验证循环 =================
        model.eval()
        val_loss = 0.0
        metrics_sum = {"dice": 0, "iou": 0, "accuracy": 0, "precision": 0, "sensitivity": 0, "specificity": 0, "hd95": 0}
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]", leave=False):
                images, masks = batch["image"].to(DEVICE), batch["mask"].to(DEVICE)
                outputs = model(images)
                
                loss = criterion(outputs, masks)
                val_loss += loss.item()
                
                batch_metrics = calculate_comprehensive_metrics(outputs, masks)
                for k in metrics_sum.keys():
                    metrics_sum[k] += batch_metrics[k]

        num_val = len(val_loader)
        avg_val_loss = val_loss / num_val
        avg_metrics = {k: v / num_val for k, v in metrics_sum.items()}

        # 保存用于画图的数据
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["val_dice"].append(avg_metrics['dice'])
        history["val_iou"].append(avg_metrics['iou'])

        # 日志打印
        log_str = (
            f"Epoch {epoch+1:03d} | LR: {current_lr:.6f} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
            f"Dice: {avg_metrics['dice']:>5.4f} | IoU: {avg_metrics['iou']:>5.4f} | "
            f"Sens: {avg_metrics['sensitivity']:>5.4f} | Spec: {avg_metrics['specificity']:>5.4f} | "
            f"Acc: {avg_metrics['accuracy']:>5.4f} | HD95: {avg_metrics['hd95']:>5.2f}"
        )
        print(log_str)
        log_file.write(log_str + "\n")
        log_file.flush()

        # 保存最优模型
        if avg_metrics['dice'] > best_dice:
            best_dice = avg_metrics['dice']
            torch.save(model.state_dict(), os.path.join(model_save_dir, "best_model.pth"))
            print(f"[*] 🚀 更新最优模型: Dice = {best_dice:.4f}")

    log_file.close()
    print(f"[*] {args.model.upper()} 训练与验证完成。验证集最高 Dice: {best_dice:.4f}")

    # ================= 5. 绘制并保存训练曲线 =================
    plt.figure(figsize=(12, 5))
    
    # 绘制 Loss 曲线
    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="Train Loss", color='blue')
    plt.plot(history["val_loss"], label="Val Loss", color='orange')
    plt.title(f"{args.model.upper()} - Loss Curve")
    plt.xlabel("Epochs")
    plt.ylabel("BCE Loss")
    plt.legend()
    plt.grid(True)

    # 绘制 Dice & IoU 曲线
    plt.subplot(1, 2, 2)
    plt.plot(history["val_dice"], label="Val Dice", color='green')
    plt.plot(history["val_iou"], label="Val IoU", color='red')
    plt.title(f"{args.model.upper()} - Metric Curve")
    plt.xlabel("Epochs")
    plt.ylabel("Score")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(model_save_dir, "training_curves.png"), dpi=300)
    plt.close()
    print(f"[*] 📊 训练曲线已保存至 {model_save_dir}/training_curves.png")

    # ================= 6. 测试集评估 (盲测) =================
    print("\n" + "="*50)
    print(f"[*] 正在加载最佳模型权重，准备在测试集上进行盲测...")
    
    best_model_path = os.path.join(model_save_dir, "best_model.pth")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=DEVICE, weights_only=True))
    else:
        print("[!] 警告：未找到最佳模型文件！")

    model.eval()
    test_metrics_sum = {"dice": 0, "iou": 0, "accuracy": 0, "precision": 0, "sensitivity": 0, "specificity": 0, "hd95": 0}
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"[*] 测试集评估 ({args.model.upper()})"):
            images, masks = batch["image"].to(DEVICE), batch["mask"].to(DEVICE)
            outputs = model(images)
            
            batch_metrics = calculate_comprehensive_metrics(outputs, masks)
            for k in test_metrics_sum.keys():
                test_metrics_sum[k] += batch_metrics[k]

    num_test = len(test_loader)
    avg_test_metrics = {k: v / num_test for k, v in test_metrics_sum.items()}

    # 打印最终测试结果
    test_result_str = (
        f"\n🏁 [FINAL TEST RESULTS] {args.model.upper()}\n"
        f"--------------------------------------------------\n"
        f"Dice:  {avg_test_metrics['dice']:.4f}\n"
        f"IoU:   {avg_test_metrics['iou']:.4f}\n"
        f"Sens:  {avg_test_metrics['sensitivity']:.4f} (Recall)\n"
        f"Spec:  {avg_test_metrics['specificity']:.4f}\n"
        f"Acc:   {avg_test_metrics['accuracy']:.4f}\n"
        f"Prec:  {avg_test_metrics['precision']:.4f}\n"
        f"HD95:  {avg_test_metrics['hd95']:.2f}\n"
        f"--------------------------------------------------"
    )
    print(test_result_str)

    # 追加写入最终测试结果
    with open(os.path.join(model_save_dir, "training_log.txt"), "a", encoding="utf-8") as f:
        f.write(test_result_str + "\n")

if __name__ == "__main__":
    main()