import os
import torch
import numpy as np
from tqdm import tqdm
from utils.metrics import dice_score, iou_score, binary_metrics


class Trainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        optimizer,
        criterion,
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
        self.save_path = os.path.join(save_dir, "best_model.pth")
        self.log_path = os.path.join(save_dir, "train.log")

        self.best_dice = 0
        self.early_stop_counter = 0

    def train(self):
        with open(self.log_path, "w") as log_file:

            for epoch in range(self.num_epochs):
                train_loss = self._train_one_epoch()
                val_metrics = self._validate()

                val_dice = val_metrics["dice"]

                log_line = (
                    f"Epoch [{epoch+1}/{self.num_epochs}] | "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Val Dice: {val_dice:.4f} | "
                    f"Val IoU: {val_metrics['iou']:.4f} | "
                    f"Val Acc: {val_metrics['accuracy']:.4f}\n"
                )

                print(log_line)
                log_file.write(log_line)
                log_file.flush()

                # Save best
                if val_dice > self.best_dice:
                    self.best_dice = val_dice
                    torch.save(self.model.state_dict(), self.save_path)
                    self.early_stop_counter = 0
                else:
                    self.early_stop_counter += 1

                if self.early_stop_counter >= self.early_stop:
                    print("Early stopping triggered.")
                    break

    def _train_one_epoch(self):
        self.model.train()
        total_loss = 0

        for batch in tqdm(self.train_loader, leave=False):
            images = batch["image"].to(self.device)
            masks = batch["mask"].unsqueeze(1).float().to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, masks)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(self.train_loader)

    def _validate(self):
        self.model.eval()

        total_dice = 0
        total_iou = 0
        total_accuracy = 0
        total_precision = 0
        total_recall = 0
        total_specificity = 0

        with torch.no_grad():
            for batch in self.val_loader:
                images = batch["image"].to(self.device)
                masks = batch["mask"].unsqueeze(1).float().to(self.device)

                outputs = self.model(images)

                total_dice += dice_score(outputs, masks)
                total_iou += iou_score(outputs, masks)

                metrics = binary_metrics(outputs, masks)
                total_accuracy += metrics["accuracy"]
                total_precision += metrics["precision"]
                total_recall += metrics["recall"]
                total_specificity += metrics["specificity"]

        n = len(self.val_loader)

        return {
            "dice": total_dice / n,
            "iou": total_iou / n,
            "accuracy": total_accuracy / n,
            "precision": total_precision / n,
            "recall": total_recall / n,
            "specificity": total_specificity / n,
        }
