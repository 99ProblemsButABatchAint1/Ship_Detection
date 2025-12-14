import numpy as np

def rle_decode(mask_rle, shape=(768, 768)):
    """
    Decodes run-length encoding into a binary mask.
    
    Args:
        mask_rle (str): Run-length as string formatted 'start length start length...'
        shape (tuple): (height, width) of array to return 
    
    Returns:
        np.ndarray: 1 - mask, 0 - background
    """
    if not isinstance(mask_rle, str):
        # Handle NaN or empty values
        return np.zeros(shape, dtype=np.uint8)
    
    s = mask_rle.split()
    # RLE format: start, length. Pairs are at even, odd indices.
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1 # Turn 1-based indexing into 0-based indexing
    ends = starts + lengths
    
    # Create flattened mask
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
        
    # Reshape to image. Note: Kaggle RLEs are usually Column-Major (Fortran style)
    # So we reshape to (W, H) then transpose to (H, W) OR reshape with order='F'
    return img.reshape(shape).T

def mask_to_rle(mask):
    """
    Encodes a binary mask into RLE string.
    
    Args:
        mask (np.ndarray): Binary mask of shape (H, W)
        
    Returns:
        str: RLE string
    """
    # Flatten column-wise
    pixels = mask.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

if __name__ == "__main__":
    # Simple round-trip test
    print("Running RLE test...")
    dummy_mask = np.zeros((768, 768), dtype=np.uint8)
    dummy_mask[100:200, 100:200] = 1 # Create a box
    
    encoded = mask_to_rle(dummy_mask)
    decoded = rle_decode(encoded)
    
    assert np.array_equal(dummy_mask, decoded), "RLE Test Failed!"
    print("RLE Test Passed!")