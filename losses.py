import torch
import torch.nn as nn
import torch.nn.functional as F

class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=0.5):
        super().__init__()
        self.bce_weight = bce_weight
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, preds, targets):
        """
        preds: Logits from the model (B, 1, H, W)
        targets: Binary ground truth masks (B, 1, H, W)
        """
        # 1. BCE Loss (Pixel-wise)
        bce_loss = self.bce(preds, targets)
        
        # 2. Dice Loss (Global overlap)
        preds_sigmoid = torch.sigmoid(preds)
        
        # Flatten tensors for Dice calculation
        preds_flat = preds_sigmoid.view(-1)
        targets_flat = targets.view(-1)
        
        intersection = (preds_flat * targets_flat).sum()
        union = preds_flat.sum() + targets_flat.sum()
        
        # Add smooth to avoid division by zero
        smooth = 1e-6
        dice_score = (2. * intersection + smooth) / (union + smooth)
        dice_loss = 1 - dice_score
        
        # Combine
        return (self.bce_weight * bce_loss) + ((1 - self.bce_weight) * dice_loss)