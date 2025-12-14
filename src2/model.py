import segmentation_models_pytorch as smp
import torch.nn as nn
from torchvision import models

# --- Stage 1: The Classifier ---
class ShipClassifier(nn.Module):
    def __init__(self, backbone='resnet34', pretrained=True):
        super().__init__()
        # Use standard torchvision ResNet
        if backbone == 'resnet34':
            self.model = models.resnet34(weights='DEFAULT' if pretrained else None)
        elif backbone == 'resnet18':
            self.model = models.resnet18(weights='DEFAULT' if pretrained else None)
        
        # Modify final layer for 1 output (Ship Probability)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 1)

    def forward(self, x):
        return self.model(x)

# --- Stage 2: The Segmenter ---
class ShipSegmenter(nn.Module):
    def __init__(self, encoder_name='resnet34', encoder_weights='imagenet', classes=1):
        super().__init__()
        self.model = smp.Unet(
            encoder_name=encoder_name, 
            encoder_weights=encoder_weights, 
            in_channels=3, 
            classes=classes, 
            activation=None
        )

    def forward(self, x):
        return self.model(x)