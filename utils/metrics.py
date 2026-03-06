import torch
import numpy as np
from scipy.ndimage import distance_transform_edt

def compute_hd95(pred, target, spacing=None):
    """
    计算 95% Hausdorff Distance (HD95)
    预测图和金标准都应该是二值化的 numpy array (0和1)
    """
    pred = np.asarray(pred, dtype=bool)
    target = np.asarray(target, dtype=bool)
    
    if pred.sum() == 0 or target.sum() == 0:
        return 0.0 # 如果全黑，根据需求也可以返回特定的惩罚值或 NaN
        
    dt_target = distance_transform_edt(~target, sampling=spacing)
    dt_pred = distance_transform_edt(~pred, sampling=spacing)
    
    # 提取边界上的距离
    sds_pred = dt_target[pred]
    sds_target = dt_pred[target]
    
    ns = sds_pred.size + sds_target.size
    if ns == 0:
        return 0.0
        
    hd95 = np.percentile(np.concatenate([sds_pred, sds_target]), 95)
    return hd95

def calculate_comprehensive_metrics(pred_logits, target_masks, smooth=1e-6):
    """
    计算全面的医学分割指标
    pred_logits: [B, 1, H, W] 网络的原始输出
    target_masks: [B, 1, H, W] 真实的二值掩码 (0.0 或 1.0)
    """
    # 转为概率并二值化
    pred_probs = torch.sigmoid(pred_logits)
    preds = (pred_probs > 0.5).float()
    targets = target_masks.float()

    # --- 区域级指标 (逐样本计算后平均) ---
    batch_size = preds.size(0)
    preds_flat = preds.view(batch_size, -1)
    targets_flat = targets.view(batch_size, -1)

    TP = (preds_flat * targets_flat).sum(dim=1)
    TN = ((1 - preds_flat) * (1 - targets_flat)).sum(dim=1)
    FP = (preds_flat * (1 - targets_flat)).sum(dim=1)
    FN = ((1 - preds_flat) * targets_flat).sum(dim=1)

    dice = (2. * TP + smooth) / (2. * TP + FP + FN + smooth)
    iou = (TP + smooth) / (TP + FP + FN + smooth)
    accuracy = (TP + TN + smooth) / (TP + TN + FP + FN + smooth)
    precision = (TP + smooth) / (TP + FP + smooth)
    sensitivity = (TP + smooth) / (TP + FN + smooth) # 等同于 Recall
    specificity = (TN + smooth) / (TN + FP + smooth)

    # --- 边界级指标 HD95 (需转到 CPU 计算) ---
    preds_np = preds.cpu().numpy()
    targets_np = targets.cpu().numpy()
    batch_hd95 = []
    
    for i in range(batch_size):
        try:
            hd_val = compute_hd95(preds_np[i, 0], targets_np[i, 0])
            batch_hd95.append(hd_val)
        except Exception:
            batch_hd95.append(0.0)
            
    avg_hd95 = np.mean(batch_hd95)

    return {
        "dice": dice.mean().item(),
        "iou": iou.mean().item(),
        "accuracy": accuracy.mean().item(),
        "precision": precision.mean().item(),
        "sensitivity": sensitivity.mean().item(),
        "specificity": specificity.mean().item(),
        "hd95": avg_hd95
    }