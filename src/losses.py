import torch
import torch.nn as nn
import torch.nn.functional as F

class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.7, smooth=1):
        """
        Tversky Loss for imbalanced datasets.
        
        Args:
            alpha (float): Weight for False Positives.
            beta (float): Weight for False Negatives.
            smooth (float): Smoothing factor to avoid division by zero.
            
        Note: 
            - alpha=0.5, beta=0.5 => Dice Coefficient
            - alpha=0.3, beta=0.7 => Emphasizes Recall (better for F2 Score)
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, preds, targets):
        # Flatten tensors
        preds = torch.sigmoid(preds).view(-1)
        targets = targets.view(-1)
        
        # True Positives, False Positives, False Negatives
        TP = (preds * targets).sum()
        FP = ((1 - targets) * preds).sum()
        FN = (targets * (1 - preds)).sum()
        
        # Tversky index
        tversky = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
        
        return 1 - tversky

class ComboLoss(nn.Module):
    def __init__(self, bce_weight=0.5, tversky_weight=0.5):
        """
        Combines BCE (for pixel stability) and Tversky (for F2 score).
        """
        super().__init__()
        self.bce_weight = bce_weight
        self.tversky_weight = tversky_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.tversky = TverskyLoss(alpha=0.3, beta=0.7)

    def forward(self, preds, targets):
        bce = self.bce(preds, targets)
        tversky = self.tversky(preds, targets)
        return (self.bce_weight * bce) + (self.tversky_weight * tversky)

# For backward compatibility if you prefer the old name
BCEDiceLoss = ComboLoss