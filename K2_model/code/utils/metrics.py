import numpy as np
import torch
from scipy.ndimage import binary_dilation, binary_erosion, distance_transform_edt
from skimage.morphology import skeletonize


METRIC_KEYS = (
    "dice",
    "iou",
    "sensitivity",
    "precision",
    "specificity",
    "accuracy",
    "hd95",
    "cldice",
    "boundary_f1",
)


def _as_bool_mask(mask):
    return np.asarray(mask).astype(bool)


def _surface(mask):
    mask = _as_bool_mask(mask)
    if mask.sum() == 0:
        return mask
    return mask ^ binary_erosion(mask)


def compute_hd95(pred, target, spacing=None, empty_value=None):
    """Compute symmetric 95th percentile Hausdorff distance on mask surfaces."""
    pred = _as_bool_mask(pred)
    target = _as_bool_mask(target)

    if pred.sum() == 0 and target.sum() == 0:
        return 0.0
    if pred.sum() == 0 or target.sum() == 0:
        return float(max(pred.shape) if empty_value is None else empty_value)

    pred_surface = _surface(pred)
    target_surface = _surface(target)
    if pred_surface.sum() == 0 or target_surface.sum() == 0:
        return float(max(pred.shape) if empty_value is None else empty_value)

    dt_target = distance_transform_edt(~target_surface, sampling=spacing)
    dt_pred = distance_transform_edt(~pred_surface, sampling=spacing)
    distances = np.concatenate([dt_target[pred_surface], dt_pred[target_surface]])
    if distances.size == 0:
        return float(empty_value)
    return float(np.percentile(distances, 95))


def cl_score(pred, target, smooth=1e-6):
    """Centerline Dice for thin/tubular structures."""
    pred = _as_bool_mask(pred)
    target = _as_bool_mask(target)
    if pred.sum() == 0 and target.sum() == 0:
        return 1.0
    if pred.sum() == 0 or target.sum() == 0:
        return 0.0

    skel_pred = skeletonize(pred)
    skel_target = skeletonize(target)
    if skel_pred.sum() == 0 or skel_target.sum() == 0:
        return 0.0

    tprec = (skel_pred & target).sum() / (skel_pred.sum() + smooth)
    tsens = (skel_target & pred).sum() / (skel_target.sum() + smooth)
    return float((2.0 * tprec * tsens) / (tprec + tsens + smooth))


def boundary_f1(pred, target, tolerance=2, smooth=1e-6):
    """Boundary F1 with a pixel tolerance around each mask surface."""
    pred = _as_bool_mask(pred)
    target = _as_bool_mask(target)
    if pred.sum() == 0 and target.sum() == 0:
        return 1.0
    if pred.sum() == 0 or target.sum() == 0:
        return 0.0

    pred_boundary = _surface(pred)
    target_boundary = _surface(target)
    if pred_boundary.sum() == 0 or target_boundary.sum() == 0:
        return 0.0

    pred_match = distance_transform_edt(~target_boundary) <= tolerance
    target_match = distance_transform_edt(~pred_boundary) <= tolerance
    precision = (pred_boundary & pred_match).sum() / (pred_boundary.sum() + smooth)
    recall = (target_boundary & target_match).sum() / (target_boundary.sum() + smooth)
    return float((2.0 * precision * recall) / (precision + recall + smooth))


def binary_metrics_from_masks(pred, target, smooth=1e-6, boundary_tolerance=2):
    """Return one per-image metric dict from binary masks."""
    pred = _as_bool_mask(pred)
    target = _as_bool_mask(target)

    tp = float(np.logical_and(pred, target).sum())
    tn = float(np.logical_and(~pred, ~target).sum())
    fp = float(np.logical_and(pred, ~target).sum())
    fn = float(np.logical_and(~pred, target).sum())

    return {
        "dice": (2.0 * tp + smooth) / (2.0 * tp + fp + fn + smooth),
        "iou": (tp + smooth) / (tp + fp + fn + smooth),
        "accuracy": (tp + tn + smooth) / (tp + tn + fp + fn + smooth),
        "precision": (tp + smooth) / (tp + fp + smooth),
        "sensitivity": (tp + smooth) / (tp + fn + smooth),
        "specificity": (tn + smooth) / (tn + fp + smooth),
        "hd95": compute_hd95(pred, target),
        "cldice": cl_score(pred, target, smooth=smooth),
        "boundary_f1": boundary_f1(pred, target, tolerance=boundary_tolerance, smooth=smooth),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def logits_to_binary_masks(pred_logits, threshold=0.5):
    probs = torch.sigmoid(pred_logits)
    return (probs > threshold).detach().cpu().numpy().astype(bool)


def target_to_binary_masks(target_masks):
    return target_masks.detach().cpu().numpy().astype(float) > 0.5


def per_image_metrics_from_logits(pred_logits, target_masks, threshold=0.5, boundary_tolerance=2):
    preds = logits_to_binary_masks(pred_logits, threshold=threshold)
    targets = target_to_binary_masks(target_masks)

    rows = []
    for i in range(preds.shape[0]):
        rows.append(
            binary_metrics_from_masks(
                preds[i, 0],
                targets[i, 0],
                boundary_tolerance=boundary_tolerance,
            )
        )
    return rows


def average_metric_rows(rows):
    if not rows:
        return {key: 0.0 for key in METRIC_KEYS}
    return {key: float(np.mean([row[key] for row in rows])) for key in METRIC_KEYS}


def calculate_comprehensive_metrics(pred_logits, target_masks, threshold=0.5, boundary_tolerance=2):
    """Backward-compatible batch-average metric API used by training scripts."""
    rows = per_image_metrics_from_logits(
        pred_logits,
        target_masks,
        threshold=threshold,
        boundary_tolerance=boundary_tolerance,
    )
    avg = average_metric_rows(rows)
    return {
        "dice": avg["dice"],
        "iou": avg["iou"],
        "accuracy": avg["accuracy"],
        "precision": avg["precision"],
        "sensitivity": avg["sensitivity"],
        "specificity": avg["specificity"],
        "hd95": avg["hd95"],
    }
