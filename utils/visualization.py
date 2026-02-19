
import matplotlib.pyplot as plt
import torch
import numpy as np

def save_comparison_grid(lr, sr, hr, filepath="results/comparison.png"):
    """
    Saves a comparison grid of LR, Bicubic (upsampled LR), SR, and HR images.
    Input Tensors: (B, C, H, W)
    """
    # Take the first 3 examples
    n = min(lr.shape[0], 3)
    
    lr = lr[:n].detach().cpu()
    sr = sr[:n].detach().cpu()
    hr = hr[:n].detach().cpu()
    
    # Upsample LR for visualization using Bicubic
    lr_big = torch.nn.functional.interpolate(lr, size=hr.shape[2:], mode='bicubic', align_corners=False)
    
    fig, axs = plt.subplots(n, 4, figsize=(12, 3*n))
    
    for i in range(n):
        # Column 1: LR (Displayed at large size but pixelated if we want, or smooth)
        # Using nearest to show pixelation
        axs[i, 0].imshow(lr[i].permute(1, 2, 0).squeeze(), cmap='inferno')
        axs[i, 0].set_title("Low Res Input")
        axs[i, 0].axis('off')
        
        # Column 2: Bicubic Baseline
        axs[i, 1].imshow(lr_big[i].permute(1, 2, 0).squeeze(), cmap='inferno')
        axs[i, 1].set_title("Bicubic Interpolation")
        axs[i, 1].axis('off')

        # Column 3: Super-Resolved
        axs[i, 2].imshow(sr[i].permute(1, 2, 0).squeeze(), cmap='inferno')
        axs[i, 2].set_title("Super-Resolved (Ours)")
        axs[i, 2].axis('off')

        # Column 4: High Res Ground Truth
        axs[i, 3].imshow(hr[i].permute(1, 2, 0).squeeze(), cmap='inferno')
        axs[i, 3].set_title("High Res Ground Truth")
        axs[i, 3].axis('off')
        
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()
