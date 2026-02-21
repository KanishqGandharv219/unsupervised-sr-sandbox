import os
import numpy as np
import argparse
from tqdm import tqdm
from scipy.ndimage import gaussian_filter

# Fixed seed for reproducibility
np.random.seed(42)

def create_mock_lens(size=64, class_type='no_sub', inner_r=8, outer_r=28, hard_mode=False):
    """
    Generates a 64x64 float32 mock image resembling DeepLense Model I format.
    - Base: Elliptical ring (SIE-like approximation)
    - PSF Blur: Gaussian (sigma 1.0 - 1.5)
    - Noise: Gaussian (SNR ~ 25)
    - Classes:
        - no_sub: Clean ring
        - vortex: Tiny circular holes/perturbations along the ring
        - subhalo: Compact bright blobs on/near the ring
    """
    img = np.zeros((size, size), dtype=np.float32)
    y, x = np.ogrid[:size, :size]
    
    # Randomize ring center slightly
    cy = size // 2 + np.random.uniform(-3, 3)
    cx = size // 2 + np.random.uniform(-3, 3)
    
    # Ellipticity
    q = np.random.uniform(0.7, 0.95)
    angle = np.random.uniform(0, np.pi)
    
    # Rotate coordinates for ellipticity
    x_rot = (x - cx) * np.cos(angle) + (y - cy) * np.sin(angle)
    y_rot = -(x - cx) * np.sin(angle) + (y - cy) * np.cos(angle)
    
    # Distance from center taking ellipticity into account
    r = np.sqrt(q * x_rot**2 + y_rot**2 / q)
    
    # Ring radius and width
    radius = np.random.uniform(inner_r, outer_r)
    width = np.random.uniform(2.0, 4.0)
    
    # Create the base ring (Gaussian-like profile radially)
    ring_brightness = np.random.uniform(0.7, 1.0)
    img += ring_brightness * np.exp(-((r - radius)**2) / (2 * (width/2)**2))
    
    # Add a central deflector lens (faint galaxy in the center)
    img += np.random.uniform(0.2, 0.5) * np.exp(-((x - cx)**2 + (y - cy)**2) / (2 * 4**2))
    
    # Apply Substructure perturbations
    if class_type == 'vortex':
        # Add 1-3 tiny dark holes along the ring
        num_vortices = np.random.randint(1, 4)
        for _ in range(num_vortices):
            v_angle = np.random.uniform(0, 2*np.pi)
            vx = cx + radius * np.cos(v_angle)
            vy = cy + radius * np.sin(v_angle)
            
            # Vortex profile (dark hole)
            v_dist = np.sqrt((x - vx)**2 + (y - vy)**2)
            v_radius = np.random.uniform(0.8, 1.5) if hard_mode else np.random.uniform(1.0, 2.5)
            v_intensity = 0.4 if hard_mode else 0.8
            # Subtract brightness
            img -= v_intensity * ring_brightness * np.exp(-(v_dist**2) / (2 * v_radius**2))
            
    elif class_type == 'subhalo':
        # Add 1-3 bright blobs near/on the ring
        num_subhalos = np.random.randint(1, 4)
        for _ in range(num_subhalos):
            s_angle = np.random.uniform(0, 2*np.pi)
            # Subhalos can be slightly perturbed off the exact radius
            s_rad = radius + np.random.uniform(-3, 3)
            sx = cx + s_rad * np.cos(s_angle)
            sy = cy + s_rad * np.sin(s_angle)
            
            s_dist = np.sqrt((x - sx)**2 + (y - sy)**2)
            s_radius = np.random.uniform(0.8, 1.8) if hard_mode else np.random.uniform(1.5, 3.0)
            s_intensity = np.random.uniform(0.2, 0.6) if hard_mode else np.random.uniform(0.5, 1.5)
            img += s_intensity * ring_brightness * np.exp(-(s_dist**2) / (2 * s_radius**2))
            
    # Clip to baseline positive physically meaningful values
    img = np.clip(img, 0, None)
    
    # Apply PSF Blurring (sigma 1.0 - 1.5) representing the physical telescope convolution
    psf_sigma = np.random.uniform(1.5, 2.0) if hard_mode else np.random.uniform(1.0, 1.5)
    img = gaussian_filter(img, sigma=psf_sigma)
    
    # Normalize clean signal between 0 and 1 before adding noise
    max_val = img.max()
    if max_val > 0:
        img /= max_val
        
    # Add Gaussian Noise (SNR ~ 25)
    # SNR = mean(signal) / std(noise).
    mean_signal = img[img > 0.1].mean() if (img > 0.1).sum() > 0 else 0.5
    snr_target = 15.0 if hard_mode else 25.0
    noise_std = mean_signal / snr_target
    noise = np.random.normal(0, noise_std, img.shape).astype(np.float32)
    img += noise
    
    img = np.clip(img, 0.0, 1.0)
    return img.astype(np.float32)

def main():
    parser = argparse.ArgumentParser(description="Generate Mock DeepLense Model I Dataset")
    parser.add_argument("--out_dir", type=str, default="data/Model_I", help="Output directory")
    parser.add_argument("--samples_per_class", type=int, default=100, help="Samples per class")
    parser.add_argument("--hard_mode", action="store_true", help="Generate a harder dataset with lower SNR and smaller structures")
    args = parser.parse_args()
    
    classes = ['no_sub', 'vortex', 'subhalo']
    
    for cls in classes:
        class_dir = os.path.join(args.out_dir, cls)
        os.makedirs(class_dir, exist_ok=True)
        
        print(f"Generating {args.samples_per_class} samples for class: {cls}")
        for i in tqdm(range(args.samples_per_class)):
            img = create_mock_lens(size=64, class_type=cls, hard_mode=args.hard_mode)
            filepath = os.path.join(class_dir, f"sample_{i:04d}.npy")
            np.save(filepath, img)
            
    print(f"\nMock dataset successfully generated at {args.out_dir}")

if __name__ == "__main__":
    main()
