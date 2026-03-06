import os
import torch
from tqdm import tqdm
from utils.metrics import dice_score, iou_score, binary_metrics

class JointTrainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        optimizer,
        criterion,  # 这里的 criterion 将是我们写的 JointDistillationLoss
        device,
        save_dir,
        num_epochs=100,
        early_stop=20
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.num_epochs = num_epochs
        self.early_stop = early_stop

        os.makedirs(save_dir, exist_ok=True)
        self.save_path = os.path.join(save_dir, "best_joint_model.pth")
        self.log_path = os.path.join(save_dir, "train.log")

        self.best_dice = 0
        self.early_stop_counter = 0

    def train(self):
        with open(self.log_path, "w") as log_file:
            for epoch in range(self.num_epochs):
                print(f"\nEpoch [{epoch+1}/{self.num_epochs}]")
                
                # 训练一个 Epoch
                train_losses = self._train_one_epoch()
                
                # 验证
                val_metrics = self._validate()

                val_dice = val_metrics["dice"]

                # 格式化日志
                log_line = (
                    f"Epoch {epoch+1} | "
                    f"Loss(Total:{train_losses['total']:.4f}, Seg:{train_losses['seg']:.4f}, "
                    f"MSE:{train_losses['mse']:.4f}, Grad:{train_losses['grad']:.4f}) | "
                    f"Val Dice: {val_dice:.4f} | "
                    f"Val IoU: {val_metrics['iou']:.4f} | "
                    f"Val Acc: {val_metrics['accuracy']:.4f}\n"
                )

                print(log_line.strip())
                log_file.write(log_line)
                log_file.flush()

                # 保存最佳模型
                if val_dice > self.best_dice:
                    self.best_dice = val_dice
                    torch.save(self.model.state_dict(), self.save_path)
                    print(f"[*] New best model saved with Dice: {self.best_dice:.4f}")
                    self.early_stop_counter = 0
                else:
                    self.early_stop_counter += 1

                if self.early_stop_counter >= self.early_stop:
                    print("=> Early stopping triggered. Training stopped.")
                    break

    def _train_one_epoch(self):
        self.model.train()
        
        # 用于累加各项指标
        epoch_losses = {"total": 0.0, "seg": 0.0, "mse": 0.0, "grad": 0.0}
        pbar = tqdm(self.train_loader, desc="Training", leave=False)

        for batch in pbar:
            images = batch["image"].to(self.device)
            masks = batch["mask"].to(self.device) # dataset 中已经 unsqueeze(0) 了
            
            # 如果提供了 teacher（训练阶段应该要有）
            teachers = batch.get("teacher")
            if teachers is not None:
                teachers = teachers.to(self.device)

            self.optimizer.zero_grad()
            
            # 联合模型推理: 返回 分割预测图 和 增强图像
            seg_preds, enhanced_imgs = self.model(images)
            
            # 计算联合损失
            if teachers is not None:
                total_loss, loss_seg, loss_mse, loss_grad = self.criterion(
                    seg_preds, masks, enhanced_imgs, teachers
                )
            else:
                # 容错处理：如果没有 teacher 也就是退化为基线模型
                total_loss, loss_seg, loss_mse, loss_grad = self.criterion(
                    seg_preds, masks, enhanced_imgs, enhanced_imgs # 假装匹配，不产生梯度
                )

            total_loss.backward()
            self.optimizer.step()

            # 记录数据
            epoch_losses["total"] += total_loss.item()
            epoch_losses["seg"] += loss_seg.item()
            epoch_losses["mse"] += loss_mse.item()
            epoch_losses["grad"] += loss_grad.item()

            pbar.set_postfix({"Total Loss": f"{total_loss.item():.4f}"})

        num_batches = len(self.train_loader)
        return {k: v / num_batches for k, v in epoch_losses.items()}

    def _validate(self):
        self.model.eval()

        total_dice, total_iou, total_acc = 0, 0, 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                images = batch["image"].to(self.device)
                masks = batch["mask"].to(self.device)

                # 推理时，我们只关心分割预测图
                seg_preds, _ = self.model(images)

                total_dice += dice_score(seg_preds, masks)
                total_iou += iou_score(seg_preds, masks)
                
                metrics = binary_metrics(seg_preds, masks)
                total_acc += metrics["accuracy"]

        n = len(self.val_loader)
        return {
            "dice": total_dice / n,
            "iou": total_iou / n,
            "accuracy": total_acc / n
        }