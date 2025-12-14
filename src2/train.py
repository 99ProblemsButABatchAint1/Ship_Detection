import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

# Import our custom modules
from dataset import AirbusDataset
from transforms import get_transforms
from model import ShipSegmenter
from losses import BCEDiceLoss
from metrics import calculate_iou, AverageMeter

# --- CONFIGURATION ---
CONFIG = {
    "TRAIN_CSV": "./data/train_split.csv",
    "VAL_CSV": "./data/val_split.csv",
    "IMG_DIR": "./data/train_v2/",
    "IMG_SIZE": 768,
    "BACKBONE": "efficientnet-b4",
    "BATCH_SIZE": 8,          # Start with 8 for 12GB VRAM. Try 10 if stable.
    "LR": 1e-4,               # Learning Rate
    "EPOCHS": 20,             # Number of passes through data
    "NUM_WORKERS": 4,         # CPU threads for data loading
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    "WEIGHT_DECAY": 1e-4,
}

def train_epoch(loader, model, criterion, optimizer, scaler, device):
    model.train()
    losses = AverageMeter()
    ious = AverageMeter()
    
    loop = tqdm(loader, leave=True)
    
    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device)
        targets = targets.to(device)

        # Forward pass with Mixed Precision
        with autocast():
            predictions = model(data)
            loss = criterion(predictions, targets)

        # Backward pass
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Metrics
        iou = calculate_iou(predictions, targets)
        
        # Update meters
        losses.update(loss.item(), data.size(0))
        ious.update(iou, data.size(0))

        # Progress bar
        loop.set_description(f"Train Loss: {losses.avg:.4f} | IoU: {ious.avg:.4f}")
        
    return losses.avg, ious.avg

def validate_epoch(loader, model, criterion, device):
    model.eval()
    losses = AverageMeter()
    ious = AverageMeter()
    
    loop = tqdm(loader, leave=True)
    
    with torch.no_grad():
        for batch_idx, (data, targets) in enumerate(loop):
            data = data.to(device)
            targets = targets.to(device)

            # No autocast needed for inference, but good practice
            with autocast():
                predictions = model(data)
                loss = criterion(predictions, targets)

            iou = calculate_iou(predictions, targets)
            
            losses.update(loss.item(), data.size(0))
            ious.update(iou, data.size(0))
            
            loop.set_description(f"Valid Loss: {losses.avg:.4f} | IoU: {ious.avg:.4f}")
            
    return losses.avg, ious.avg

def main():
    print(f"--- Starting Training on {CONFIG['DEVICE']} ---")
    
    # 1. Load Data
    train_df = pd.read_csv(CONFIG['TRAIN_CSV'])
    val_df = pd.read_csv(CONFIG['VAL_CSV'])
    
    # 2. FILTER: Train only on images with ships (Stage 2 Strategy)
    print(f"Original Train Size: {len(train_df)}")
    train_df = train_df[train_df['HasShip'] == 1].reset_index(drop=True)
    # We also filter val set to measure segmentation performance specifically
    val_df = val_df[val_df['HasShip'] == 1].reset_index(drop=True)
    print(f"Filtered (Ship Only) Train Size: {len(train_df)}")
    print(f"Filtered (Ship Only) Val Size: {len(val_df)}")

    # 3. Datasets & Loaders
    train_dataset = AirbusDataset(
        df=train_df,
        image_dir=CONFIG['IMG_DIR'],
        transform=get_transforms(phase='train', img_size=CONFIG['IMG_SIZE'])
    )
    
    val_dataset = AirbusDataset(
        df=val_df,
        image_dir=CONFIG['IMG_DIR'],
        transform=get_transforms(phase='valid', img_size=CONFIG['IMG_SIZE'])
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG['BATCH_SIZE'],
        num_workers=CONFIG['NUM_WORKERS'],
        pin_memory=True,
        shuffle=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG['BATCH_SIZE'],
        num_workers=CONFIG['NUM_WORKERS'],
        pin_memory=True,
        shuffle=False
    )
    
    # 4. Model Setup
    model = ShipSegmenter(encoder_name=CONFIG['BACKBONE']).to(CONFIG['DEVICE'])
    
    # 5. Optimization
    loss_fn = BCEDiceLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['LR'], weight_decay=CONFIG['WEIGHT_DECAY'])
    scaler = GradScaler() # Enables mixed precision
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['EPOCHS'], eta_min=1e-6)
    
    # 6. Loop
    best_loss = float('inf')
    
    for epoch in range(CONFIG['EPOCHS']):
        print(f"\nEpoch {epoch+1}/{CONFIG['EPOCHS']}")
        
        train_loss, train_iou = train_epoch(train_loader, model, loss_fn, optimizer, scaler, CONFIG['DEVICE'])
        val_loss, val_iou = validate_epoch(val_loader, model, loss_fn, CONFIG['DEVICE'])
        
        scheduler.step()
        
        print(f"Epoch {epoch+1} Summary:")
        print(f"Train Loss: {train_loss:.4f} | Train IoU: {train_iou:.4f}")
        print(f"Valid Loss: {val_loss:.4f} | Valid IoU: {val_iou:.4f}")
        
        # Save Best Model
        if val_loss < best_loss:
            best_loss = val_loss
            print(f"Validation Loss Improved. Saving model...")
            torch.save(model.state_dict(), "best_ship_segmenter.pth")
            
    print("Training Complete!")

if __name__ == "__main__":
    main()