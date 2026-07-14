import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class GradientLoss(nn.Module):
    """Sobel gradient consistency on the green channel of enhanced/teacher images."""

    def __init__(self):
        super().__init__()
        kernel_x = [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
        kernel_y = [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]
        self.register_buffer("kernel_x", torch.FloatTensor(kernel_x).unsqueeze(0).unsqueeze(0))
        self.register_buffer("kernel_y", torch.FloatTensor(kernel_y).unsqueeze(0).unsqueeze(0))

    def forward(self, pred, target):
        pred_g = pred[:, 1:2, :, :]
        target_g = target[:, 1:2, :, :]
        grad_x_pred = F.conv2d(pred_g, self.kernel_x, padding=1)
        grad_y_pred = F.conv2d(pred_g, self.kernel_y, padding=1)
        grad_x_gt = F.conv2d(target_g, self.kernel_x, padding=1)
        grad_y_gt = F.conv2d(target_g, self.kernel_y, padding=1)
        return F.mse_loss(grad_x_pred, grad_x_gt) + F.mse_loss(grad_y_pred, grad_y_gt)


class BCEDiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.smooth = smooth

    def forward(self, logits, target):
        loss_bce = self.bce_loss(logits, target)
        probs = torch.sigmoid(logits)
        intersection = (probs * target).sum()
        dice_loss = 1.0 - (2.0 * intersection + self.smooth) / (
            probs.sum() + target.sum() + self.smooth
        )
        return loss_bce + dice_loss


class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.7, gamma=0.75, smooth=1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, logits, target):
        probs = torch.sigmoid(logits)
        dims = tuple(range(1, probs.ndim))
        tp = (probs * target).sum(dim=dims)
        fp = (probs * (1.0 - target)).sum(dim=dims)
        fn = ((1.0 - probs) * target).sum(dim=dims)
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return torch.pow(1.0 - tversky, self.gamma).mean()


class UnifiedFocalLoss(nn.Module):
    """Lightweight unified focal loss: focal BCE + focal Tversky."""

    def __init__(self, alpha=0.75, beta=0.7, gamma=0.75, smooth=1e-6):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.focal_tversky = FocalTverskyLoss(alpha=1.0 - beta, beta=beta, gamma=gamma, smooth=smooth)

    def forward(self, logits, target):
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        pt = torch.exp(-bce)
        alpha_factor = target * self.alpha + (1.0 - target) * (1.0 - self.alpha)
        focal_bce = (alpha_factor * torch.pow(1.0 - pt, self.gamma) * bce).mean()
        return 0.5 * focal_bce + 0.5 * self.focal_tversky(logits, target)


def _soft_erode(img):
    if img.shape[2] <= 2 or img.shape[3] <= 2:
        return img
    p1 = -F.max_pool2d(-img, kernel_size=(3, 1), stride=1, padding=(1, 0))
    p2 = -F.max_pool2d(-img, kernel_size=(1, 3), stride=1, padding=(0, 1))
    return torch.min(p1, p2)


def _soft_dilate(img):
    return F.max_pool2d(img, kernel_size=3, stride=1, padding=1)


def _soft_open(img):
    return _soft_dilate(_soft_erode(img))


def _soft_skeletonize(img, iterations=10):
    img1 = _soft_open(img)
    skel = F.relu(img - img1)
    for _ in range(iterations):
        img = _soft_erode(img)
        img1 = _soft_open(img)
        delta = F.relu(img - img1)
        skel = skel + F.relu(delta - skel * delta)
    return skel


class SoftClDiceLoss(nn.Module):
    def __init__(self, iterations=10, smooth=1e-6):
        super().__init__()
        self.iterations = iterations
        self.smooth = smooth

    def forward(self, logits, target):
        probs = torch.sigmoid(logits)
        skel_pred = _soft_skeletonize(probs, iterations=self.iterations)
        skel_target = _soft_skeletonize(target, iterations=self.iterations)
        tprec = (skel_pred * target).sum() / (skel_pred.sum() + self.smooth)
        tsens = (skel_target * probs).sum() / (skel_target.sum() + self.smooth)
        cldice = (2.0 * tprec * tsens + self.smooth) / (tprec + tsens + self.smooth)
        return 1.0 - cldice


class BoundaryDiceLoss(nn.Module):
    """Differentiable boundary Dice loss using soft erosion boundaries."""

    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, target):
        probs = torch.sigmoid(logits)
        pred_boundary = F.relu(probs - _soft_erode(probs))
        target_boundary = F.relu(target - _soft_erode(target))
        intersection = (pred_boundary * target_boundary).sum()
        denom = pred_boundary.sum() + target_boundary.sum()
        return 1.0 - (2.0 * intersection + self.smooth) / (denom + self.smooth)


class CenterlineBoundaryDiceLoss(nn.Module):
    """Soft centerline-boundary Dice for vascular structures.

    This approximates the cbDice idea with differentiable skeleton and boundary maps:
    centerline pixels are rewarded when they fall inside the target/predicted vessel,
    while soft boundaries keep vessel width from drifting too much.
    """

    def __init__(self, iterations=10, smooth=1e-6):
        super().__init__()
        self.iterations = iterations
        self.smooth = smooth

    def forward(self, logits, target):
        probs = torch.sigmoid(logits)
        skel_pred = _soft_skeletonize(probs, iterations=self.iterations)
        skel_target = _soft_skeletonize(target, iterations=self.iterations)
        pred_boundary = F.relu(probs - _soft_erode(probs))
        target_boundary = F.relu(target - _soft_erode(target))

        center_precision = (skel_pred * target).sum() / (skel_pred.sum() + self.smooth)
        center_sensitivity = (skel_target * probs).sum() / (skel_target.sum() + self.smooth)
        boundary_overlap = (pred_boundary * target_boundary).sum() / (pred_boundary.sum() + target_boundary.sum() + self.smooth)

        cbdice = (2.0 * center_precision * center_sensitivity + self.smooth) / (
            center_precision + center_sensitivity + self.smooth
        )
        cbdice = 0.7 * cbdice + 0.3 * (2.0 * boundary_overlap)
        return 1.0 - cbdice.clamp(0.0, 1.0)


class CompositeSegmentationLoss(nn.Module):
    def __init__(self, base_loss, cldice_weight=0.0, boundary_weight=0.0, cbdice_weight=0.0):
        super().__init__()
        self.base_loss = base_loss
        self.cldice_weight = cldice_weight
        self.boundary_weight = boundary_weight
        self.cbdice_weight = cbdice_weight
        self.cldice_loss = SoftClDiceLoss() if cldice_weight > 0 else None
        self.boundary_loss = BoundaryDiceLoss() if boundary_weight > 0 else None
        self.cbdice_loss = CenterlineBoundaryDiceLoss() if cbdice_weight > 0 else None

    def forward(self, logits, target):
        loss = self.base_loss(logits, target)
        if self.cldice_loss is not None:
            loss = loss + self.cldice_weight * self.cldice_loss(logits, target)
        if self.boundary_loss is not None:
            loss = loss + self.boundary_weight * self.boundary_loss(logits, target)
        if self.cbdice_loss is not None:
            loss = loss + self.cbdice_weight * self.cbdice_loss(logits, target)
        return loss

def build_segmentation_loss(
    seg_loss="bce_dice",
    cldice_weight=0.5,
    boundary_weight=0.5,
    cbdice_weight=0.5,
    focal_alpha=0.3,
    focal_beta=0.7,
    focal_gamma=0.75,
):
    if seg_loss == "bce_dice":
        return BCEDiceLoss()
    if seg_loss == "focal_tversky":
        return FocalTverskyLoss(alpha=focal_alpha, beta=focal_beta, gamma=focal_gamma)
    if seg_loss == "unified_focal":
        return UnifiedFocalLoss(alpha=focal_alpha, beta=focal_beta, gamma=focal_gamma)
    if seg_loss == "bce_dice_cldice":
        return CompositeSegmentationLoss(BCEDiceLoss(), cldice_weight=cldice_weight, boundary_weight=0.0, cbdice_weight=0.0)
    if seg_loss == "bce_dice_boundary":
        return CompositeSegmentationLoss(BCEDiceLoss(), cldice_weight=0.0, boundary_weight=boundary_weight, cbdice_weight=0.0)
    if seg_loss == "bce_dice_cldice_boundary":
        return CompositeSegmentationLoss(
            BCEDiceLoss(),
            cldice_weight=cldice_weight,
            boundary_weight=boundary_weight,
            cbdice_weight=0.0,
        )
    if seg_loss == "bce_dice_cbdice":
        return CompositeSegmentationLoss(BCEDiceLoss(), cldice_weight=0.0, boundary_weight=0.0, cbdice_weight=cbdice_weight)
    if seg_loss == "bce_dice_cbdice_boundary":
        return CompositeSegmentationLoss(BCEDiceLoss(), cldice_weight=0.0, boundary_weight=boundary_weight, cbdice_weight=cbdice_weight)
    raise ValueError(f"Unknown seg_loss: {seg_loss}")


class JointDistillationLoss(nn.Module):
    def __init__(
        self,
        lambda_mse=10.0,
        lambda_grad=30.0,
        weight_mode="fixed",
        seg_loss="bce_dice",
        cldice_weight=0.5,
        boundary_weight=0.5,
        cbdice_weight=0.5,
        focal_alpha=0.3,
        focal_beta=0.7,
        focal_gamma=0.75,
    ):
        super().__init__()
        self.seg_loss = build_segmentation_loss(
            seg_loss=seg_loss,
            cldice_weight=cldice_weight,
            boundary_weight=boundary_weight,
            cbdice_weight=cbdice_weight,
            focal_alpha=focal_alpha,
            focal_beta=focal_beta,
            focal_gamma=focal_gamma,
        )
        self.mse_loss = nn.MSELoss()
        self.grad_loss = GradientLoss()
        self.lambda_mse = lambda_mse
        self.lambda_grad = lambda_grad
        self.weight_mode = weight_mode

        if weight_mode == "learnable":
            if lambda_mse <= 0 or lambda_grad <= 0:
                raise ValueError(
                    "learnable loss weighting requires positive lambda_mse and lambda_grad "
                    f"(got lambda_mse={lambda_mse}, lambda_grad={lambda_grad})"
                )
            self.log_var_mse = nn.Parameter(torch.tensor(-math.log(lambda_mse), dtype=torch.float32))
            self.log_var_grad = nn.Parameter(torch.tensor(-math.log(lambda_grad), dtype=torch.float32))
        elif weight_mode != "fixed":
            raise ValueError(f"Unknown weight_mode: {weight_mode}")

    def get_distill_weights(self):
        if self.weight_mode == "learnable":
            return {
                "mse": torch.exp(-self.log_var_mse).detach().item(),
                "grad": torch.exp(-self.log_var_grad).detach().item(),
            }
        return {"mse": float(self.lambda_mse), "grad": float(self.lambda_grad)}

    def forward(self, seg_pred, mask_target, enhanced_img, teacher_img):
        total_seg_loss = self.seg_loss(seg_pred, mask_target)
        loss_mse = self.mse_loss(enhanced_img, teacher_img)
        loss_grad = self.grad_loss(enhanced_img, teacher_img)

        if self.weight_mode == "learnable":
            weight_mse = torch.exp(-self.log_var_mse)
            weight_grad = torch.exp(-self.log_var_grad)
            total_loss = (
                total_seg_loss
                + weight_mse * loss_mse
                + self.log_var_mse
                + weight_grad * loss_grad
                + self.log_var_grad
            )
        else:
            total_loss = total_seg_loss + self.lambda_mse * loss_mse + self.lambda_grad * loss_grad

        return total_loss, total_seg_loss, loss_mse, loss_grad

def soft_boundary_target(mask):
    return F.relu(mask - _soft_erode(mask))


class JointDistillationBoundaryLoss(JointDistillationLoss):
    def __init__(self, boundary_aux_weight=0.3, **kwargs):
        super().__init__(**kwargs)
        self.boundary_aux_weight = boundary_aux_weight
        self.boundary_aux_loss = BCEDiceLoss()

    def forward(self, seg_pred, mask_target, enhanced_img, teacher_img, boundary_pred):
        total_loss, total_seg_loss, loss_mse, loss_grad = super().forward(
            seg_pred,
            mask_target,
            enhanced_img,
            teacher_img,
        )
        boundary_target = soft_boundary_target(mask_target)
        loss_boundary_aux = self.boundary_aux_loss(boundary_pred, boundary_target)
        total_loss = total_loss + self.boundary_aux_weight * loss_boundary_aux
        return total_loss, total_seg_loss, loss_mse, loss_grad, loss_boundary_aux


class JointDecoderDistillationLoss(JointDistillationLoss):
    """Image-level prior distillation plus decoder feature consistency."""

    def __init__(
        self,
        lambda_decoder_distill=1.0,
        decoder_distill_layers="2,3",
        decoder_distill_mode="mse",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.lambda_decoder_distill = lambda_decoder_distill
        self.decoder_distill_layers = self._parse_layers(decoder_distill_layers)
        self.decoder_distill_mode = decoder_distill_mode

    @staticmethod
    def _parse_layers(layers):
        if isinstance(layers, (list, tuple)):
            return [int(layer) for layer in layers]
        parsed = []
        for item in str(layers).split(","):
            item = item.strip()
            if item:
                parsed.append(int(item))
        return parsed

    def _single_feature_loss(self, student_feature, teacher_feature):
        if student_feature.shape[-2:] != teacher_feature.shape[-2:]:
            teacher_feature = F.interpolate(
                teacher_feature,
                size=student_feature.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        if student_feature.shape[1] != teacher_feature.shape[1]:
            raise ValueError(
                f"Decoder feature channel mismatch: student={student_feature.shape[1]} "
                f"teacher={teacher_feature.shape[1]}"
            )

        if self.decoder_distill_mode == "mse":
            return F.mse_loss(student_feature, teacher_feature)

        student_norm = F.normalize(student_feature, dim=1)
        teacher_norm = F.normalize(teacher_feature, dim=1)
        normalized_mse = F.mse_loss(student_norm, teacher_norm)
        cosine_loss = 1.0 - F.cosine_similarity(student_norm, teacher_norm, dim=1).mean()

        if self.decoder_distill_mode == "normalized_mse":
            return normalized_mse
        if self.decoder_distill_mode == "cosine":
            return cosine_loss
        if self.decoder_distill_mode == "cosine_mse":
            return 0.5 * normalized_mse + 0.5 * cosine_loss
        raise ValueError(f"Unknown decoder_distill_mode: {self.decoder_distill_mode}")

    def _feature_distill_loss(self, feature_pair):
        if feature_pair is None:
            device = self.lambda_mse.device if torch.is_tensor(self.lambda_mse) else None
            return torch.tensor(0.0, device=device)
        if isinstance(feature_pair, dict):
            student_features = feature_pair["student_features"]
            teacher_features = feature_pair["teacher_features"]
        else:
            student_features, teacher_features = feature_pair
        if len(student_features) != len(teacher_features):
            raise ValueError(
                f"Decoder feature count mismatch: student={len(student_features)} teacher={len(teacher_features)}"
            )
        selected = self.decoder_distill_layers or list(range(len(student_features)))
        losses = []
        for layer_idx in selected:
            if layer_idx < 0 or layer_idx >= len(student_features):
                raise ValueError(f"decoder_distill layer {layer_idx} out of range 0..{len(student_features)-1}")
            losses.append(self._single_feature_loss(student_features[layer_idx], teacher_features[layer_idx]))
        if not losses:
            return student_features[-1].new_tensor(0.0)
        return torch.stack(losses).mean()

    def forward(self, seg_pred, mask_target, enhanced_img, teacher_img, decoder_feature_pair):
        total_loss, total_seg_loss, loss_mse, loss_grad = super().forward(
            seg_pred,
            mask_target,
            enhanced_img,
            teacher_img,
        )
        loss_decoder = self._feature_distill_loss(decoder_feature_pair)
        total_loss = total_loss + self.lambda_decoder_distill * loss_decoder
        return total_loss, total_seg_loss, loss_mse, loss_grad, loss_decoder
