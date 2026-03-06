import os
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from datasets.dataset_vessel import VesselDataset
from utils.metrics import calculate_comprehensive_metrics

# 导入创新模块
from models.joint_framework import Enhancer, JointModel
from losses.joint_loss import JointDistillationLoss
from models.transunet import TransUNet

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./dataset_raw_split")
    parser.add_argument("--save_dir", type=str, default="./experiments/ours_transunet")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    # 蒸馏权重的超参数
    parser.add_argument("--lambda_mse", type=float, default=5.0, help="强度蒸馏权重")
    parser.add_argument("--lambda_grad", type=float, default=5.0, help="边缘梯度蒸馏权重")
    return parser.parse_args()

def main():
    args = get_args()
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.save_dir, exist_ok=True)
    log_file = open(os.path.join(args.save_dir, "training_log.txt"), "w", encoding="utf-8")

    print(f"[*] 启动 OURS (Joint Distillation) 训练... | 设备: {DEVICE}")

    # ================= 1. 数据加载 (引入 Teacher) =================
    train_dataset = VesselDataset(
        image_dir=os.path.join(args.data_dir, "train/images"),
        mask_dir=os.path.join(args.data_dir, "train/masks"),
        teacher_dir=os.path.join(args.data_dir, "train/teacher_priors"), # 必须有！
        img_size=256, augment=True
    )
    # 验证集和测试集不需要 Teacher，只看最终分割效果
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

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # ================= 2. 模型与优化器配置 =================
    enhancer = Enhancer(in_channels=3, out_channels=3)
    # 使用我们修复好的 512 通道级 TransUNet 作为分割器
    segmentor = TransUNet(n_channels=3, n_classes=1, img_size=256) 
    
    model = JointModel(enhancer, segmentor).to(DEVICE)

    criterion = JointDistillationLoss(lambda_mse=args.lambda_mse, lambda_grad=args.lambda_grad).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    # 写入超参数日志
    log_file.write(f"=== OURS (TransUNet Backbone) ===\nMSE_Wt: {args.lambda_mse}, Grad_Wt: {args.lambda_grad}\n\n")
    best_dice = 0.0
    history = {"train_loss": [], "val_dice": [], "val_iou": [], "val_hd95": []}

    # ================= 3. 训练循环 =================
    for epoch in range(args.epochs):
        model.train()
        train_loss, train_seg, train_mse, train_grad = 0.0, 0.0, 0.0, 0.0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]", leave=False):
            images, masks = batch["image"].to(DEVICE), batch["mask"].to(DEVICE)
            teachers = batch["teacher"].to(DEVICE)
            
            optimizer.zero_grad()
            seg_preds, enhanced_imgs = model(images)
            
            # 计算联合损失
            loss, l_seg, l_mse, l_grad = criterion(seg_preds, masks, enhanced_imgs, teachers)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_seg += l_seg.item()
            train_mse += l_mse.item()
            train_grad += l_grad.item()

        N = len(train_loader)
        avg_loss = train_loss / N
        scheduler.step()

        # ================= 4. 验证循环 =================
        model.eval()
        metrics_sum = {"dice": 0, "iou": 0, "accuracy": 0, "hd95": 0}
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]", leave=False):
                images, masks = batch["image"].to(DEVICE), batch["mask"].to(DEVICE)
                # 推理时，解包拿出 seg_out 即可，不需要管 enhanced_imgs
                seg_preds, _ = model(images) 
                
                batch_metrics = calculate_comprehensive_metrics(seg_preds, masks)
                for k in metrics_sum.keys():
                    metrics_sum[k] += batch_metrics[k]

        avg_metrics = {k: v / len(val_loader) for k, v in metrics_sum.items()}
        history["train_loss"].append(avg_loss)
        history["val_dice"].append(avg_metrics['dice'])
        history["val_hd95"].append(avg_metrics['hd95'])

        # 日志打印 (加上子 Loss 的监控，这对 Debug 非常重要)
        log_str = (
            f"Ep {epoch+1:03d} | Loss(Tot:{avg_loss:.3f} Seg:{train_seg/N:.3f} MSE:{train_mse/N:.3f} Grad:{train_grad/N:.3f}) | "
            f"Val Dice: {avg_metrics['dice']:.4f} | HD95: {avg_metrics['hd95']:.2f}"
        )
        print(log_str)
        log_file.write(log_str + "\n")
        log_file.flush()

        if avg_metrics['dice'] > best_dice:
            best_dice = avg_metrics['dice']
            torch.save(model.state_dict(), os.path.join(args.save_dir, "best_model.pth"))
            print(f"[*] 🚀 更新最优模型: Dice = {best_dice:.4f}")

    log_file.close()

    # ================= 5. 测试集评估 (盲测) =================
    print("\n" + "="*50)
    print(f"[*] 加载最佳模型权重，准备盲测...")
    model.load_state_dict(torch.load(os.path.join(args.save_dir, "best_model.pth"), map_location=DEVICE, weights_only=True))
    model.eval()
    
    test_metrics_sum = {"dice": 0, "iou": 0, "accuracy": 0, "precision": 0, "sensitivity": 0, "specificity": 0, "hd95": 0}
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="[*] OURS 测试集评估"):
            images, masks = batch["image"].to(DEVICE), batch["mask"].to(DEVICE)
            outputs, _ = model(images)
            batch_metrics = calculate_comprehensive_metrics(outputs, masks)
            for k in test_metrics_sum.keys(): test_metrics_sum[k] += batch_metrics[k]

    avg_test_metrics = {k: v / len(test_loader) for k, v in test_metrics_sum.items()}

    test_result_str = (
        f"\n🏁 [FINAL TEST RESULTS] OURS (Enhancer + TransUNet + Distillation)\n"
        f"--------------------------------------------------\n"
        f"Dice:  {avg_test_metrics['dice']:.4f}\n"
        f"HD95:  {avg_test_metrics['hd95']:.2f}\n"
        f"Sens:  {avg_test_metrics['sensitivity']:.4f} (Recall)\n"
        f"Spec:  {avg_test_metrics['specificity']:.4f}\n"
        f"IoU:   {avg_test_metrics['iou']:.4f}\n"
        f"Acc:   {avg_test_metrics['accuracy']:.4f}\n"
        f"Prec:  {avg_test_metrics['precision']:.4f}\n"  # <--- 补上这一行
        f"--------------------------------------------------"
    )
    print(test_result_str)
    with open(os.path.join(args.save_dir, "training_log.txt"), "a", encoding="utf-8") as f:
        f.write(test_result_str + "\n")

if __name__ == "__main__":
    main()