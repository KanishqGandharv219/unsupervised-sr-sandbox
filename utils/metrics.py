
import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim
import math

def calculate_psnr(sr, hr, max_val=1.0):
    """
    Calculate PSNR (Peak Signal-to-Noise Ratio).
    sr, hr: Tensors (B, C, H, W) or (C, H, W)
    """
    mse = torch.mean((sr - hr) ** 2)
    if mse == 0:
        return 100.0
    return 20 * math.log10(max_val) - 10 * math.log10(torch.sqrt(mse).item())

def calculate_ssim(sr, hr, data_range=1.0):
    """
    Calculate SSIM (Structural Similarity Index) using scikit-image.
    Note: Converts tensors to numpy. This is slow for large batches.
    """
    # Detach and move to cpu
    sr_np = sr.detach().cpu().numpy()
    hr_np = hr.detach().cpu().numpy()
    
    # Simple loop over batch
    batch_size = sr_np.shape[0]
    total_ssim = 0.0
    
    for i in range(batch_size):
        # ssim expects (H, W, C) or (H, W)
        # Our images are (C, H, W). Since C=1, we can squeeze.
        img1 = sr_np[i].squeeze()
        img2 = hr_np[i].squeeze()
        
        total_ssim += ssim(img1, img2, data_range=data_range)
        
    return total_ssim / batch_size
