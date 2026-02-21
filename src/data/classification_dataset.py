import os
from glob import glob
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class DeepLenseClassificationDataset(Dataset):
    def __init__(self, root_dir, mode='bicubic', sr_model_path=None, split='train'):
        self.mode = mode
        self.split = split
        self.sr_model = None
        
        # Fixed seed for reproducibility across train/val splits
        np.random.seed(42)
        
        # Load file paths and labels
        self.samples = []
        for label, class_name in enumerate(['no_sub', 'vortex', 'subhalo']):
            class_dir = os.path.join(root_dir, class_name)
            files = sorted(glob(f'{class_dir}/*.npy'))
            
            # Shuffle deterministically before splitting
            files = np.array(files)
            np.random.shuffle(files)
            files = files.tolist()
            
            # 80/20 train/val split is stratified by class
            split_idx = int(0.8 * len(files))
            files = files[:split_idx] if split == 'train' else files[split_idx:]
            
            for f in files:
                self.samples.append((f, label))
                
        # Data Augmentation for training split only
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
        ]) if split == 'train' else None
                
        # Load SR model if needed
        if mode == 'sr_hybrid' and sr_model_path:
            import sys
            # Assuming this will be run from project root, ensure models can be imported
            sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
            from models.baseline import SimpleSRNet
            
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.sr_model = SimpleSRNet().to(self.device)
            self.sr_model.load_state_dict(torch.load(sr_model_path, map_location=self.device))
            self.sr_model.eval()
            
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        hr_img = np.load(path)  # (64, 64)
        
        if self.mode == 'hr':
            img = torch.FloatTensor(hr_img).unsqueeze(0)  # (1, 64, 64)
            
        elif self.mode == 'bicubic':
            # Downsample to 32x32, then bicubic upsample back to 64x64
            lr = cv2.resize(hr_img, (32, 32), interpolation=cv2.INTER_AREA)
            img_bicubic = cv2.resize(lr, (64, 64), interpolation=cv2.INTER_CUBIC)
            img = torch.FloatTensor(img_bicubic).unsqueeze(0)
            
        elif self.mode == 'sr_hybrid':
            # Downsample to 32x32, run through SR model
            lr = cv2.resize(hr_img, (32, 32), interpolation=cv2.INTER_AREA)
            lr_tensor = torch.FloatTensor(lr).unsqueeze(0).unsqueeze(0).to(self.device)  # (1, 1, 32, 32)
            
            with torch.no_grad():
                sr_output = self.sr_model(lr_tensor).clamp(0, 1)  # (1, 1, 64, 64)
            img = sr_output.squeeze(0).cpu()  # (1, 64, 64)
            
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
            
        if self.transform:
            img = self.transform(img)
            
        return img, label
    
    def __len__(self):
        return len(self.samples)
