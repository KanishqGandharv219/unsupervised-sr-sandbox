
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
from utils.metrics import calculate_psnr, calculate_ssim
from utils.visualization import save_comparison_grid

def total_variation_loss(img):
    """
    Compute Total Variation (TV) regularization.
    Encourages spatial smoothness while preserving edges.
    """
    # Horizontal differences
    diff_h = torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :])
    # Vertical differences  
    diff_w = torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1])
    
    tv = diff_h.mean() + diff_w.mean()
    return tv

class Trainer:
    def __init__(self, model, train_loader, val_loader, device='cuda', lr=1e-3, results_dir='results', physics=None, mode='hybrid', lambda_tv=0.01):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.results_dir = results_dir
        self.physics = physics # PhysicsDownsampler instance
        
        self.mode = mode
        self.lambda_tv = lambda_tv
        
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        os.makedirs(results_dir, exist_ok=True)
        
    def train_epoch(self, epoch, lambda_physics=0.0):
        self.model.train()
        running_loss = 0.0
        running_sup_loss = 0.0
        running_phy_loss = 0.0
        running_tv_loss = 0.0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]")
        
        for batch in pbar:
            lr_imgs, hr_imgs = batch[0].to(self.device), batch[1].to(self.device)
            
            # Zero grad
            self.optimizer.zero_grad()
            
            # Forward
            sr_imgs = self.model(lr_imgs)
            
            total_loss = torch.tensor(0.0, device=self.device)
            
            # 1. Supervised Loss (SR vs HR)
            sup_loss = torch.tensor(0.0, device=self.device)
            if self.mode in ['supervised', 'hybrid']:
                sup_loss = self.criterion(sr_imgs, hr_imgs)
                # If mode is explicitly supervised, we don't multiply by a lambda
                lambda_sup = 1.0
                total_loss += lambda_sup * sup_loss
            
            # 2. Physics/Unsupervised Loss (P(SR) vs LR)
            phy_loss = torch.tensor(0.0, device=self.device)
            if self.mode in ['hybrid', 'unsupervised'] and self.physics is not None:
                # Generate Simulated LR from SR (Deterministic)
                lr_pred = self.physics(sr_imgs, deterministic=True)
                
                if lr_pred.shape != lr_imgs.shape:
                    lr_pred = torch.nn.functional.interpolate(lr_pred, size=lr_imgs.shape[2:], mode='bilinear', align_corners=False)
                
                phy_loss = self.criterion(lr_pred, lr_imgs)
                # In unsupervised mode, physics is the primary driver (lambda_physics typically 1.0)
                # In hybrid, it's weighted by lambda_physics
                total_loss += lambda_physics * phy_loss
                
            # 3. Total Variation Regularization (Unsupervised Only)
            tv_loss = torch.tensor(0.0, device=self.device)
            if self.mode == 'unsupervised':
                tv_loss = total_variation_loss(sr_imgs)
                total_loss += self.lambda_tv * tv_loss
            
            # Backward
            total_loss.backward()
            self.optimizer.step()
            
            running_loss += total_loss.item()
            running_sup_loss += sup_loss.item()
            running_phy_loss += phy_loss.item()
            running_tv_loss += tv_loss.item()
            
            pbar.set_postfix({
                'loss': total_loss.item(), 
                'sup': sup_loss.item(), 
                'phy': phy_loss.item(),
                'tv': tv_loss.item()
            })
            
        return running_loss / len(self.train_loader)
    
    def validate(self, epoch):
        self.model.eval()
        running_psnr = 0.0
        running_ssim = 0.0
        
        # Save viz for first batch
        saved_viz = False
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc=f"Epoch {epoch} [Val]"):
                lr_imgs, hr_imgs = batch[0].to(self.device), batch[1].to(self.device)
                
                sr_imgs = self.model(lr_imgs).clamp(0, 1) # Ensure valid range for legacy
                
                # Psmnr/Ssim
                batch_psnr = calculate_psnr(sr_imgs, hr_imgs)
                batch_ssim = calculate_ssim(sr_imgs, hr_imgs) # CPU bottleneck potential
                
                running_psnr += batch_psnr
                running_ssim += batch_ssim
                
                if not saved_viz:
                    save_comparison_grid(lr_imgs, sr_imgs, hr_imgs, 
                                         filepath=f"{self.results_dir}/epoch_{epoch}_viz.png")
                    saved_viz = True
                    
        avg_psnr = running_psnr / len(self.val_loader)
        avg_ssim = running_ssim / len(self.val_loader)
        
        return avg_psnr, avg_ssim

    def fit(self, epochs=10, lambda_physics=0.0):
        best_psnr = 0.0
        
        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(epoch, lambda_physics=lambda_physics)
            val_psnr, val_ssim = self.validate(epoch)
            
            print(f"Epoch {epoch}: Train Loss: {train_loss:.5f} | Val PSNR: {val_psnr:.2f} | Val SSIM: {val_ssim:.4f}")
            
            # Save checkpoints
            if val_psnr > best_psnr:
                best_psnr = val_psnr
                torch.save(self.model.state_dict(), f"{self.results_dir}/best_model.pth")
                print(f"Saved Best Model (PSNR: {best_psnr:.2f})")
            
            # Always save metrics to a log file
            with open(f"{self.results_dir}/training_log.txt", "a") as f:
                f.write(f"{epoch},{train_loss},{val_psnr},{val_ssim}\n")
