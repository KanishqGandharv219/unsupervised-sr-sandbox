
import torch
import numpy as np
from torch.utils.data import Dataset
from utils.physics import PhysicsDownsampler
import random

class SyntheticLensingDataset(Dataset):
    """
    Generates synthetic "lensing-like" images on the fly.
    Features:
    - Dark background
    - Gaussian blobs (lenses/sources)
    - Arcs (distorted sources)
    """
    def __init__(self, num_samples=1000, img_size=64, scale_factor=2):
        self.num_samples = num_samples
        self.img_size = img_size
        self.physics = PhysicsDownsampler(scale_factor=scale_factor)
        
    def __len__(self):
        return self.num_samples
    
    def _draw_gaussian(self, img, center, sigma, amplitude):
        y, x = np.ogrid[:self.img_size, :self.img_size]
        cy, cx = center
        mask = np.exp(-((x - cx)**2 + (y - cy)**2) / (2 * sigma**2))
        img += mask * amplitude
        return img

    def _draw_arc(self, img, center, radius, width, start_angle, end_angle, intensity):
        # Very simple arc approximation using ring sector
        y, x = np.ogrid[:self.img_size, :self.img_size]
        cy, cx = center
        
        # Radial distance matrix
        r2 = (x - cx)**2 + (y - cy)**2
        
        # Angle matrix
        theta = np.arctan2(y - cy, x - cx)
        
        # Mask for the ring
        ring_mask = (r2 >= (radius - width/2)**2) & (r2 <= (radius + width/2)**2)
        
        # Mask for the angle
        # Handling wrap around pi/-pi is tricky, doing simplified check
        if start_angle < end_angle:
            angle_mask = (theta >= start_angle) & (theta <= end_angle)
        else:
             angle_mask = (theta >= start_angle) | (theta <= end_angle)
             
        img[ring_mask & angle_mask] += intensity
        return img

    def __getitem__(self, idx):
        # 1. Generate HR Image (numpy)
        img = np.zeros((self.img_size, self.img_size), dtype=np.float32)
        
        # Randomize parameters
        # Add a "Lens" (central blob)
        if random.random() > 0.2:
            self._draw_gaussian(img, 
                                center=(self.img_size//2, self.img_size//2), 
                                sigma=random.uniform(3, 8), 
                                amplitude=random.uniform(0.5, 1.0))
        
        # Add "Source/Arc"
        num_arcs = random.randint(1, 3)
        for _ in range(num_arcs):
            is_arc = random.random() > 0.3
            if is_arc:
                self._draw_arc(img,
                               center=(self.img_size//2, self.img_size//2),
                               radius=random.uniform(10, self.img_size//3),
                               width=random.uniform(1, 3),
                               start_angle=random.uniform(-3, 3),
                               end_angle=random.uniform(-3, 3),
                               intensity=random.uniform(0.5, 1.0))
            else:
                # Just a blob source
                cx = random.randint(0, self.img_size)
                cy = random.randint(0, self.img_size)
                self._draw_gaussian(img, (cy, cx), random.uniform(2, 5), random.uniform(0.3, 0.8))
        
        # Add some background noise
        img += np.random.normal(0, 0.05, img.shape)
        img = np.clip(img, 0, 1)
        
        # Convert to Tensor (1, H, W)
        hr_tensor = torch.from_numpy(img).unsqueeze(0).float()
        
        # 2. Physics Downsampling -> LR Image
        # Note: PhysicsDownsampler expects (B, C, H, W) usually, but handles 3D
        lr_tensor = self.physics(hr_tensor)
        
        # Squeeze the batch dim if physics added it? No, physics returns (..., H, W)
        if lr_tensor.dim() == 4:
            lr_tensor = lr_tensor.squeeze(0)
            
        return lr_tensor, hr_tensor
