import os
import cv2
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from model import ShipSegmenter
from transforms import get_transforms
from rle import mask_to_rle

# --- CONFIG ---
# Update this path to point to your trained weights
SEGMENTER_PATH = "best_ship_segmenter.pth" 
TEST_DIR = "./data/test_v2/"
OUTPUT_FILE = "submission.csv"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# IMPORTANT: Update this with the value you found in Step 3 (tune_thresholds.py)
THRESHOLD = 0.5    
MIN_AREA = 50      # Remove ship detections smaller than 50 pixels

def load_model():
    """Loads the trained segmentation model."""
    model = ShipSegmenter(encoder_name='efficientnet-b4')
    try:
        model.load_state_dict(torch.load(SEGMENTER_PATH, map_location=DEVICE))
        print(f"Loaded Segmenter from {SEGMENTER_PATH}")
    except FileNotFoundError:
        print(f"Warning: Weights file {SEGMENTER_PATH} not found.")
    
    model.to(DEVICE)
    model.eval()
    return model

def predict_with_tta(model, img_tensor):
    """
    Test-Time Augmentation (TTA): 
    Predicts on original, horizontally flipped, and vertically flipped images.
    Averages the results to smooth out errors and improve F2 score.
    """
    # 1. Predict on Original Image
    logits = model(img_tensor)
    pred_orig = torch.sigmoid(logits)
    
    # 2. Horizontal Flip
    # Flip input (dim 3 is width) -> Predict -> Flip output back
    img_hflip = torch.flip(img_tensor, dims=[3])
    logits_h = model(img_hflip)
    pred_h = torch.flip(torch.sigmoid(logits_h), dims=[3])
    
    # 3. Vertical Flip
    # Flip input (dim 2 is height) -> Predict -> Flip output back
    img_vflip = torch.flip(img_tensor, dims=[2])
    logits_v = model(img_vflip)
    pred_v = torch.flip(torch.sigmoid(logits_v), dims=[2])
    
    # Average the predictions
    return (pred_orig + pred_h + pred_v) / 3.0

def post_process(mask, min_area=50):
    """
    Remove small blobs (noise) from the binary mask.
    """
    # Apply Threshold
    mask = (mask > THRESHOLD).astype(np.uint8)
    
    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
    
    cleaned_mask = np.zeros_like(mask)
    
    # Filter by area size (stats[i, 4] is the area of the blob)
    for i in range(1, num_labels): # Start from 1 to skip background (0)
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            cleaned_mask[labels == i] = 1
            
    return cleaned_mask

def run_inference():
    model = load_model()
    transforms = get_transforms(phase='valid')
    
    # Get all test images
    if not os.path.exists(TEST_DIR):
        print(f"Error: Test directory {TEST_DIR} not found.")
        return

    test_images = os.listdir(TEST_DIR)
    results = []
    
    print(f"Running TTA Inference on {len(test_images)} test images...")
    print(f"Using Threshold: {THRESHOLD} | Min Area: {MIN_AREA}")
    
    for img_name in tqdm(test_images):
        img_path = os.path.join(TEST_DIR, img_name)
        
        # 1. Load & Preprocess
        image = cv2.imread(img_path)
        if image is None: continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply transforms (Normalize + ToTensor)
        augmented = transforms(image=image)
        img_tensor = augmented['image'].unsqueeze(0).to(DEVICE)
        
        # 2. Model Prediction with TTA
        with torch.no_grad():
            # Use our new TTA function instead of a single model call
            probs = predict_with_tta(model, img_tensor)
            
        # 3. Post-Processing
        pred_mask = probs.squeeze().cpu().numpy()
        clean_mask = post_process(pred_mask, min_area=MIN_AREA)
        
        # 4. Encode
        rle = mask_to_rle(clean_mask)
        results.append({'ImageId': img_name, 'EncodedPixels': rle})
        
    # Save Submission
    sub_df = pd.DataFrame(results)
    sub_df.to_csv(OUTPUT_FILE, index=False)
    print(f"Submission saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    run_inference()