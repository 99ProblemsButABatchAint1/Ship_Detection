import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from rle import rle_decode

class AirbusDataset(Dataset):
    def __init__(self, df, image_dir, transform=None):
        self.df = df
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_id = row['ImageId']
        rle_list = row['RleMasks'] 
        
        img_path = os.path.join(self.image_dir, image_id)
        image = cv2.imread(img_path)
        
        if image is None:
            image = np.zeros((768, 768, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = np.zeros((768, 768), dtype=np.float32)
        
        if isinstance(rle_list, list):
            for rle in rle_list:
                if isinstance(rle, str) and len(rle) > 0:
                    decoded = rle_decode(rle)
                    mask += decoded
        
        mask = np.clip(mask, 0, 1)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
            
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
            
        return image, mask