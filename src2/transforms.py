import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_transforms(phase='train', img_size=768):
    """
    Returns Albumentations transforms for training or validation.
    
    Args:
        phase (str): 'train' or 'valid'
        img_size (int): Target image size (default 768 for Airbus)
    """
    if phase == 'train':
        return A.Compose([
            # --- Geometric Augmentations ---
            # Ships look the same from any angle (rotation invariant)
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            
            # Slight affine shifts to make the model robust to positioning
            A.ShiftScaleRotate(
                shift_limit=0.05, 
                scale_limit=0.05, 
                rotate_limit=15, 
                p=0.5
            ),
            
            # --- Pixel-level Augmentations ---
            # Simulate different sea/weather conditions
            A.OneOf([
                A.RandomBrightnessContrast(p=1),
                A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=1),
                A.GaussNoise(p=1),
                A.Blur(blur_limit=3, p=1),
            ], p=0.3),
            
            # --- Normalization ---
            # Standard ImageNet normalization for pretrained models
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
        
    else:
        # Validation: No augmentation, just normalization
        return A.Compose([
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])