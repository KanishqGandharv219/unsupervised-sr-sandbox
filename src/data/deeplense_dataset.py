import os
import torch
import numpy as np
from torch.utils.data import Dataset

class DeepLenseDataset(Dataset):
    """
    PyTorch Dataset for DeepLense Model I/II/III simulated images.
    
    Expected directory structure:
        data/
          Model_I/
            no_sub/       # No substructure
            vortex/       # Axion vortex substructure  
            subhalo/      # CDM subhalo substructure
          Model_II/
            ...
            
    Returns:
        (lr, hr, label) tuple where:
        - lr: [1, lr_size, lr_size] tensor (if physics downsampler is provided)
        - hr: [1, hr_size, hr_size] tensor  
        - label: int (0=no_sub, 1=vortex, 2=subhalo)
    """
    def __init__(self, root_dir, model='Model_I', classes=None, split='train', 
                 lr_size=32, hr_size=64, physics_downsampler=None, transform=None):
        if classes is None:
            classes = ['no_sub', 'vortex', 'subhalo']
        
        self.root_dir = root_dir
        self.model = model
        self.classes = classes
        self.split = split
        self.lr_size = lr_size
        self.hr_size = hr_size
        self.physics = physics_downsampler
        self.transform = transform
        
        # Build file list
        self.samples = []
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        
        base_dir = os.path.join(root_dir, model)
        if not os.path.exists(base_dir):
            raise FileNotFoundError(f"DeepLense directory not found at {base_dir}. Please run data generator or download datasets.")
            
        for cls in self.classes:
            cls_dir = os.path.join(base_dir, cls)
            if not os.path.exists(cls_dir):
                print(f"Warning: Class directory {cls_dir} not found. Skipping.")
                continue
                
            files = [f for f in os.listdir(cls_dir) if f.endswith('.npy') or f.endswith('.fits')]
            # Sort for deterministic splitting
            files.sort()
            
            # Simple 80/10/10 split
            total_files = len(files)
            train_end = int(0.8 * total_files)
            val_end = int(0.9 * total_files)
            
            if split == 'train':
                split_files = files[:train_end]
            elif split == 'val':
                split_files = files[train_end:val_end]
            elif split == 'test':
                split_files = files[val_end:]
            else:
                split_files = files # use all
                
            for file in split_files:
                self.samples.append((os.path.join(cls_dir, file), self.class_to_idx[cls]))
                
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        file_path, label = self.samples[idx]
        
        # Load image (assuming .npy for now)
        if file_path.endswith('.npy'):
            img_np = np.load(file_path).astype(np.float32)
        else:
            # Fallback if fits is added later
            img_np = np.zeros((self.hr_size, self.hr_size), dtype=np.float32)
            
        # Ensure correct shape (C, H, W)
        if img_np.ndim == 2:
            img_np = np.expand_dims(img_np, axis=0) # [1, H, W]
            
        if self.transform:
            # Simple custom transforms could be applied here
            pass
            
        hr_tensor = torch.from_numpy(img_np)
        
        # Apply physics downsampler to generate LR if available
        if self.physics is not None:
            # Physics downsampler needs [B, C, H, W], so add batch dim then remove
            hr_batch = hr_tensor.unsqueeze(0)
            
            # Move to same device as physics simulator if needed
            if hasattr(self.physics, 'device'):
                hr_batch = hr_batch.to(self.physics.device)
                
            lr_batch = self.physics(hr_batch, deterministic=True)
            lr_tensor = lr_batch.squeeze(0).cpu()
        else:
            # Fallback downsample
            lr_tensor = torch.nn.functional.interpolate(hr_tensor.unsqueeze(0), size=(self.lr_size, self.lr_size), mode='bicubic', align_corners=False).squeeze(0)
            
        return lr_tensor, hr_tensor, label
