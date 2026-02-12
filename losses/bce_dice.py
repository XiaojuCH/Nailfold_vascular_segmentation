import torch
import torch.nn as nn


class BCEDiceLoss(nn.Module):
    def __init__(self):
        super(BCEDiceLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, pred, target):
        bce = self.bce(pred, target)

        pred = torch.sigmoid(pred)
        smooth = 1e-6

        intersection = (pred * target).sum()
        dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

        dice_loss = 1 - dice

        return bce + dice_loss
