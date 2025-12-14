import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from ast import literal_eval

# Imports from your project
from dataset import AirbusDataset
from transforms import get_transforms
from model import ShipSegmenter
from metrics import calculate_f2_score

# --- CONFIG ---
MODEL_PATH = "best_ship_segmenter.pth"
VAL_CSV = "./data/val_split.csv"
IMG_DIR = "./data/train_v2/"
IMG_SIZE = 768
BATCH_SIZE = 16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_data():
    """Loads and preprocesses validation data"""
    val_df = pd.read_csv(VAL_CSV)
    
    # 1. Preprocess RLE strings to lists
    print("Preprocessing dataframe...")
    if len(val_df) > 0 and isinstance(val_df['RleMasks'].iloc[0], str):
        val_df['RleMasks'] = val_df['RleMasks'].apply(literal_eval)
    
    # 2. Filter for evaluation (Only ships? Or All?)
    # For accurate Threshold Tuning, it's best to check purely on images WITH ships
    # to see how well the segmentation works.
    val_df = val_df[val_df['HasShip'] == 1].reset_index(drop=True)
    
    dataset = AirbusDataset(
        df=val_df,
        image_dir=IMG_DIR,
        transform=get_transforms(phase='valid', img_size=IMG_SIZE)
    )
    
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=4, shuffle=False)
    return loader

def find_best_threshold(model, loader):
    print("Running inference to gather predictions...")
    
    # Store all probabilities and targets to CPU to sweep thresholds quickly
    all_probs = []
    all_targets = []
    
    model.eval()
    with torch.no_grad():
        for data, targets in tqdm(loader):
            data = data.to(DEVICE)
            
            # Get probabilities
            logits = model(data)
            probs = torch.sigmoid(logits).cpu() # Move to CPU to save GPU memory
            
            all_probs.append(probs)
            all_targets.append(targets.cpu())
            
    # Concatenate all batches
    all_probs = torch.cat(all_probs)
    all_targets = torch.cat(all_targets)
    
    print("\n--- Starting Threshold Sweep (0.10 - 0.90) ---")
    best_score = 0.0
    best_thresh = 0.5
    
    # Check thresholds from 0.1 to 0.9 in steps of 0.05
    thresholds = np.arange(0.1, 0.95, 0.05)
    
    for thresh in thresholds:
        # Calculate F2 for the entire dataset at this threshold
        f2 = calculate_f2_score(all_probs, all_targets, threshold=thresh)
        print(f"Threshold {thresh:.2f} | F2 Score: {f2:.4f}")
        
        if f2 > best_score:
            best_score = f2
            best_thresh = thresh
            
    print("\n" + "="*30)
    print(f"✅ BEST RESULT:")
    print(f"Optimal Threshold: {best_thresh:.2f}")
    print(f"Max F2 Score:      {best_score:.4f}")
    print("="*30)
    
    return best_thresh

if __name__ == "__main__":
    # 1. Setup
    model = ShipSegmenter(encoder_name='efficientnet-b4')
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print(f"Loaded model from {MODEL_PATH}")
    except FileNotFoundError:
        print("Error: Model weights not found. Train the model first!")
        exit()
        
    model.to(DEVICE)
    
    # 2. Data
    val_loader = load_data()
    
    # 3. Tuning
    find_best_threshold(model, val_loader)