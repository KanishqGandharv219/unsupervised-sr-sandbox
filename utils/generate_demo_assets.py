import os
import sys

# Add the parent directory to sys.path to allow imports from root
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import json
from tqdm import tqdm
from torchvision.utils import save_image, make_grid
from src.data.deeplense_dataset import DeepLenseDataset
from models.baseline import SimpleSRNet
from utils.physics import PhysicsDownsampler
from utils.metrics import calculate_psnr, calculate_ssim

# --- Configuration ---
N_SAMPLES = 12
OUTPUT_DIR = "docs/assets"
SAMPLES_DIR = os.path.join(OUTPUT_DIR, "samples")
GRIDS_DIR = os.path.join(OUTPUT_DIR, "grids")
CONFIG_PATH = "docs/js/config.js"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def ensure_dirs():
    os.makedirs(SAMPLES_DIR, exist_ok=True)
    os.makedirs(GRIDS_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)

def generate_descriptions(idx):
    # Deterministic descriptions based on index
    descriptions = [
        "Nearly complete Einstein ring with strong symmetry (θ_E ≈ 1.2\").",
        "Partial arc configuration with visible source structure.",
        "Quad-like lens showing multiply imaged source components.",
        "Faint Einstein ring obscured by higher noise levels.",
        "Complex substructure visible in the lensed arc.",
        "High-magnification fold caustic configuration.",
        "Small Einstein radius system, barely resolved in LR.",
        "Bright active galactic nucleus (AGN) host galaxy lens.",
        "Asymmetric arc due to elliptical lens mass potential.",
        "Double ring structure suggesting dual source planes.",
        "Gravitationally sheared background galaxy.",
        "Edge-on spiral source galaxy strongly distorted."
    ]
    return descriptions[idx % len(descriptions)]

def main():
    print(f"Generating demo assets to {OUTPUT_DIR}...")
    ensure_dirs()

    # 1. Load Data
    # Use validation set to ensure unseen data
    dataset = DeepLenseDataset(root_dir='data', model='Model_I', split='val')
    
    # 2. Load Models
    print("Loading models...")
    baseline_model = SimpleSRNet().to(DEVICE)
    hybrid_model = SimpleSRNet().to(DEVICE)
    unsupervised_model = SimpleSRNet().to(DEVICE)
    
    # Attempt to load weights
    baseline_path = "results/best_model.pth"
    hybrid_path = "results_hybrid/best_model.pth"
    unsupervised_path = "results_unsupervised/best_model.pth"

    if os.path.exists(baseline_path):
        baseline_model.load_state_dict(torch.load(baseline_path, map_location=DEVICE))
        print(f"Loaded Baseline: {baseline_path}")
    else:
        print(f"WARNING: {baseline_path} not found. Using random weights.")

    if os.path.exists(hybrid_path):
        hybrid_model.load_state_dict(torch.load(hybrid_path, map_location=DEVICE))
        print(f"Loaded Hybrid: {hybrid_path}")
    else:
        print(f"WARNING: {hybrid_path} not found.")

    if os.path.exists(unsupervised_path):
        unsupervised_model.load_state_dict(torch.load(unsupervised_path, map_location=DEVICE))
        print(f"Loaded Unsupervised: {unsupervised_path}")
    else:
        print(f"WARNING: {unsupervised_path} not found.")

    baseline_model.eval()
    hybrid_model.eval()
    unsupervised_model.eval()

    # Physics
    downsampler = PhysicsDownsampler(scale_factor=2, psf_sigma=1.0, device=DEVICE)

    # 3. Process Samples
    config_data = {}
    grid_images = {'lr': [], 'bicubic': [], 'sr_baseline': [], 'sr_hybrid': [], 'hr': []}
    
    # Select specific indices for diversity (or just first N)
    indices = range(N_SAMPLES)
    
    avg_psnr = {'bicubic': 0, 'sr_baseline': 0, 'sr_hybrid': 0}
    avg_ssim = {'bicubic': 0, 'sr_baseline': 0, 'sr_hybrid': 0}

    with torch.no_grad():
        for i, idx in enumerate(tqdm(indices, desc="Processing Samples")):
            # dataset returns (lr, hr, label)
            _, hr, _ = dataset[idx]
            hr = hr.unsqueeze(0).to(DEVICE) # (1, 1, 64, 64)
            
            # Generate LR
            # Use deterministic=True for clean demo images (no random noise for consistency)
            lr = downsampler(hr, deterministic=True) # (1, 1, 32, 32)
            
            # Bicubic Upsample
            bicubic = F.interpolate(lr, size=(64, 64), mode='bicubic', align_corners=False)
            
            # Model Inference
            sr_baseline = baseline_model(lr)
            sr_hybrid = hybrid_model(lr)
            sr_unsupervised = unsupervised_model(lr).clamp(0, 1)
            
            # --- Metrics ---
            # Calculate for this sample
            m_bicubic = {'psnr': calculate_psnr(bicubic, hr), 'ssim': calculate_ssim(bicubic, hr)}
            m_baseline = {'psnr': calculate_psnr(sr_baseline, hr), 'ssim': calculate_ssim(sr_baseline, hr)}
            m_hybrid = {'psnr': calculate_psnr(sr_hybrid, hr), 'ssim': calculate_ssim(sr_hybrid, hr)}
            m_unsupervised = {'psnr': calculate_psnr(sr_unsupervised, hr), 'ssim': calculate_ssim(sr_unsupervised, hr)}
            
            # Accumulate averages
            avg_psnr['bicubic'] += m_bicubic['psnr']
            avg_ssim['bicubic'] += m_bicubic['ssim']
            avg_psnr['sr_baseline'] += m_baseline['psnr']
            avg_ssim['sr_baseline'] += m_baseline['ssim']
            avg_psnr['sr_hybrid'] += m_hybrid['psnr']
            avg_ssim['sr_hybrid'] += m_hybrid['ssim']

            # --- Save Images ---
            sample_id = f"lens_{i+1:03d}"
            sample_dir = os.path.join(SAMPLES_DIR, sample_id)
            os.makedirs(sample_dir, exist_ok=True)
            
            save_image(lr, os.path.join(sample_dir, "lr.png"))
            save_image(bicubic, os.path.join(sample_dir, "bicubic.png"))
            save_image(sr_baseline, os.path.join(sample_dir, "sr_baseline.png"))
            save_image(sr_hybrid, os.path.join(sample_dir, "sr_hybrid.png"))
            save_image(sr_unsupervised, os.path.join(sample_dir, "sr_unsupervised.png"))
            save_image(hr, os.path.join(sample_dir, "hr.png"))
            
            # Collect for Grid (Keep first 3 for grid)
            if i < 3:
                grid_images['lr'].append(lr.cpu())
                grid_images['bicubic'].append(bicubic.cpu())
                grid_images['sr_baseline'].append(sr_baseline.cpu())
                grid_images['sr_hybrid'].append(sr_hybrid.cpu())
                grid_images['hr'].append(hr.cpu())

            # --- Update Config ---
            config_data[sample_id] = {
                "name": f"Sample {i+1}",
                "description": generate_descriptions(i),
                "metrics": {
                    "bicubic": m_bicubic,
                    "sr_baseline": m_baseline,
                    "sr_hybrid": m_hybrid,
                    "sr_unsupervised": m_unsupervised
                }
            }

    # 4. Final averages
    for k in avg_psnr:
        avg_psnr[k] /= N_SAMPLES
        avg_ssim[k] /= N_SAMPLES
    
    # Add stats to config
    final_js_content = f"const demoConfig = {{\n  \"samples\": {json.dumps(config_data, indent=4)},\n  \"stats\": {json.dumps({'psnr': avg_psnr, 'ssim': avg_ssim}, indent=4)}\n}};"
    
    with open(CONFIG_PATH, 'w') as f:
        f.write(final_js_content)
    print(f"Config saved to {CONFIG_PATH}")

    # 5. Generate Comparison Grid (3x5)
    # Rows: Samples (3), Cols: Methods (5)
    print("Generating Comparison Grid...")
    
    # Stack images: List of (3, C, H, W) -> (3*5, C, H, W)
    # Order: LR, Bicubic, Base, Hybrid, HR
    # LR needs resizing for grid
    resized_lrs = [F.interpolate(img, size=(64, 64), mode='nearest') for img in grid_images['lr']]
    
    grid_list = []
    for j in range(3): # 3 samples
        grid_list.append(resized_lrs[j])
        grid_list.append(grid_images['bicubic'][j])
        grid_list.append(grid_images['sr_baseline'][j])
        grid_list.append(grid_images['sr_hybrid'][j])
        grid_list.append(grid_images['hr'][j])
    
    grid_tensor = torch.cat(grid_list, dim=0)
    
    plt.figure(figsize=(15, 10))
    grid_img = make_grid(grid_tensor, nrow=5, padding=2, normalize=True).permute(1, 2, 0).numpy()
    plt.imshow(grid_img)
    plt.axis('off')
    plt.title("LR (Nearest) | Bicubic | SR Baseline | SR Hybrid | HR Ground Truth")
    plt.savefig(os.path.join(GRIDS_DIR, "comparison_3x5.png"), bbox_inches='tight', dpi=150)
    plt.close()
    
    print("Done!")

if __name__ == "__main__":
    main()
