import torch
import numpy as np

def calculate_f2_score(preds, targets, threshold=0.5, epsilon=1e-7):
    """
    Calculates the F2 Score (beta=2), which weighs Recall higher than Precision.
    
    Args:
        preds: Logits tensor (B, 1, H, W) or Probabilities
        targets: Binary mask tensor (B, 1, H, W)
        threshold: Threshold to convert logits/probs to binary
    """
    with torch.no_grad():
        # Check if preds are logits (contain negative values) or probs [0,1]
        if preds.min() < 0:
            preds = torch.sigmoid(preds)
            
        preds_bin = (preds > threshold).float()
        targets_bin = targets.float()
        
        # Flatten
        preds_flat = preds_bin.view(-1)
        targets_flat = targets_bin.view(-1)
        
        tp = (preds_flat * targets_flat).sum().item()
        fp = (preds_flat * (1 - targets_flat)).sum().item()
        fn = ((1 - preds_flat) * targets_flat).sum().item()
        
        precision = tp / (tp + fp + epsilon)
        recall = tp / (tp + fn + epsilon)
        
        # F2 Score Formula: (5 * P * R) / (4 * P + R)
        f2 = (5 * precision * recall) / (4 * precision + recall + epsilon)
        
        return f2

def calculate_iou(preds, targets, threshold=0.5):
    """
    Calculates Intersection over Union.
    """
    with torch.no_grad():
        # Handle logits vs probs
        if preds.min() < 0:
            preds = torch.sigmoid(preds)
            
        preds_bin = (preds > threshold).byte()
        targets_bin = targets.byte()
        
        preds_flat = preds_bin.view(-1)
        targets_flat = targets_bin.view(-1)
        
        intersection = (preds_flat & targets_flat).sum().item()
        union = (preds_flat | targets_flat).sum().item()
        
        if union == 0:
            return 1.0
            
        return intersection / union

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count