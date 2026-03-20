import torch
import torch.nn as nn
import torch.nn.functional as F

class GradientLoss(nn.Module):
    """边缘梯度约束 (基于 Sobel 算子)"""
    def __init__(self):
        super(GradientLoss, self).__init__()
        kernel_x = [[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]
        kernel_y = [[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]
        self.kernel_x = torch.FloatTensor(kernel_x).unsqueeze(0).unsqueeze(0).cuda()
        self.kernel_y = torch.FloatTensor(kernel_y).unsqueeze(0).unsqueeze(0).cuda()

    def forward(self, pred, target):
        pred_g = pred[:, 1:2, :, :]
        target_g = target[:, 1:2, :, :]
        grad_x_pred = F.conv2d(pred_g, self.kernel_x, padding=1)
        grad_y_pred = F.conv2d(pred_g, self.kernel_y, padding=1)
        grad_x_gt = F.conv2d(target_g, self.kernel_x, padding=1)
        grad_y_gt = F.conv2d(target_g, self.kernel_y, padding=1)
        return F.mse_loss(grad_x_pred, grad_x_gt) + F.mse_loss(grad_y_pred, grad_y_gt)

class JointDistillationLoss(nn.Module):
    def __init__(self, lambda_mse=5.0, lambda_grad=5.0):
        super(JointDistillationLoss, self).__init__()
        # 回归最纯粹的 BCE，和 Baseline 保持绝对公平
        self.bce_loss = nn.BCEWithLogitsLoss() 
        self.mse_loss = nn.MSELoss()
        self.grad_loss = GradientLoss()
        
        # 把 lambda_mse 和 lambda_grad 提高到 20.0
        self.lambda_mse = 10.0 
        self.lambda_grad = 30.0 

    def forward(self, seg_pred, mask_target, enhanced_img, teacher_img):
        # 1. 基础分割 Loss (BCE + Dice)
        loss_bce = self.bce_loss(seg_pred, mask_target)
        
        pred_sig = torch.sigmoid(seg_pred)
        intersection = (pred_sig * mask_target).sum()
        dice_loss = 1 - (2. * intersection + 1e-6) / (pred_sig.sum() + mask_target.sum() + 1e-6)
        
        total_seg_loss = loss_bce + dice_loss
        
        # 2. 物理先验蒸馏 Loss
        loss_mse = self.mse_loss(enhanced_img, teacher_img)
        loss_grad = self.grad_loss(enhanced_img, teacher_img)
        
        # 3. 联合总 Loss
        total_loss = total_seg_loss + self.lambda_mse * loss_mse + self.lambda_grad * loss_grad
        
        return total_loss, total_seg_loss, loss_mse, loss_grad