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
MODEL_PATH = "best_ship_segmenter.pth"
TEST_DIR = "./data/test_v2/"
OUTPUT_FILE = "submission.csv"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
THRESHOLD = 0.5    # Pixel probability threshold
MIN_AREA = 50      # Remove ship detections smaller than 50 pixels

def load_model():
    model = ShipSegmenter(encoder_name='efficientnet-b4')
    model.load_state_dict(torch.load(MODEL_PATH))
    model.to(DEVICE)
    model.eval()
    return model

def post_process(mask, min_area=50):
    """
    Remove small blobs (noise) from the mask.
    """
    mask = (mask > THRESHOLD).astype(np.uint8)
    
    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
    
    cleaned_mask = np.zeros_like(mask)
    
    # Filter by area size (stats[i, 4] is area)
    for i in range(1, num_labels): # Start from 1 to skip background
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            cleaned_mask[labels == i] = 1
            
    return cleaned_mask

def run_inference():
    model = load_model()
    transforms = get_transforms(phase='valid')
    
    # Get all test images
    test_images = os.listdir(TEST_DIR)
    results = []
    
    print(f"Running inference on {len(test_images)} test images...")
    
    for img_name in tqdm(test_images):
        img_path = os.path.join(TEST_DIR, img_name)
        
        # 1. Load & Preprocess
        image = cv2.imread(img_path)
        if image is None: continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply transforms (Normalize + ToTensor)
        augmented = transforms(image=image)
        img_tensor = augmented['image'].unsqueeze(0).to(DEVICE)
        
        # 2. Model Prediction (Stage 2)
        # Note: In a full pipeline, you would run the Classifier (Stage 1) here first.
        # If classifier_score < 0.5: prediction = ""
        # Else: run segmenter
        
        with torch.no_grad():
            logits = model(img_tensor)
            probs = torch.sigmoid(logits)
            
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