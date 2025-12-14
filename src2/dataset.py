import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from .rle import rle_decode

class AirbusDataset(Dataset):
    def __init__(self, df, image_dir, transform=None):
        """
        Args:
            df (pd.DataFrame): DataFrame containing 'ImageId' and 'RleMasks'
            image_dir (str): Path to the directory containing images
            transform (albumentations.Compose): Augmentation pipeline
        """
        self.df = df
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # 1. Get row data
        row = self.df.iloc[idx]
        image_id = row['ImageId']
        rle_list = row['RleMasks'] # This is a list of strings
        
        # 2. Load Image
        img_path = os.path.join(self.image_dir, image_id)
        image = cv2.imread(img_path)
        
        # Safety check for missing images
        if image is None:
            # Return a blank entry or raise error depending on preference.
            # For training stability, creating a blank placeholder is safer than crashing.
            print(f"Warning: Image not found {img_path}")
            image = np.zeros((768, 768, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 3. Generate Mask
        # Start with a black mask
        mask = np.zeros((768, 768), dtype=np.float32)
        
        # Handle the case where RleMasks is loaded as a string "['...']" by pandas
        if isinstance(rle_list, str):
            import ast
            try:
                rle_list = ast.literal_eval(rle_list)
            except:
                rle_list = []
                
        # If lists exists and is not empty (and not just a single NaN), decode
        if isinstance(rle_list, list) and len(rle_list) > 0:
            for rle in rle_list:
                if isinstance(rle, str) and len(rle) > 0:
                    decoded = rle_decode(rle)
                    mask += decoded
        
        # Clip mask to [0, 1]. 
        # Overlapping ships in RLEs are additive, but binary mask should be 0 or 1.
        mask = np.clip(mask, 0, 1)

        # 4. Apply Augmentations
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
            
        # 5. Format for PyTorch
        # Image is already Tensor via ToTensorV2 in transform
        # Mask needs channel dimension if it doesn't have one: (H, W) -> (1, H, W)
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
            
        return image, mask