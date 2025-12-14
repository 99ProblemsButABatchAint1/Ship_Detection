import torch
import numpy as np

def calculate_iou(preds, targets, threshold=0.5):
    """
    Calculates intersection over union for a batch.
    
    Args:
        preds: Logits tensor (B, 1, H, W)
        targets: Binary mask tensor (B, 1, H, W)
        threshold: Threshold to convert logits to binary
    """
    with torch.no_grad():
        preds_sigmoid = torch.sigmoid(preds)
        preds_bin = (preds_sigmoid > threshold).byte()
        targets_bin = targets.byte()
        
        # Flatten
        preds_flat = preds_bin.view(-1)
        targets_flat = targets_bin.view(-1)
        
        intersection = (preds_flat & targets_flat).sum().item()
        union = (preds_flat | targets_flat).sum().item()
        
        if union == 0:
            # Avoid division by zero. 
            # If union is 0, both pred and target are empty -> IoU is 1.0
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