import os
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader, random_split
from data.lensing_dataset import SyntheticLensingDataset
from src.data.deeplense_dataset import DeepLenseDataset
from models.baseline import SimpleSRNet
from training.trainer import Trainer
from utils.physics import PhysicsDownsampler

def main():
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    np.random.seed(42)

    parser = argparse.ArgumentParser(description="DeepLense-SR Pure Unsupervised Training")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--samples", type=int, default=1000, help="Number of synthetic samples")
    parser.add_argument("--dataset", type=str, choices=['synthetic', 'deeplense'], default='deeplense', help="Dataset type to use")
    parser.add_argument("--deeplense_model", type=str, choices=['Model_I', 'Model_II', 'Model_III'], default='Model_I', help="DeepLense model to load (if dataset=deeplense)")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--lambda_phy", type=float, default=1.0, help="Weight for physics loss")
    parser.add_argument("--lambda_tv", type=float, default=0.0001, help="Weight for Total Variation loss")
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Dataset & Dataloaders
    physics = PhysicsDownsampler(scale_factor=2, device=device)
    
    if args.dataset == 'synthetic':
        print("Generating Synthetic Lensing Data...")
        dataset = SyntheticLensingDataset(num_samples=args.samples, scale_factor=2)
        
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_ds, val_ds = random_split(dataset, [train_size, val_size])
        
    elif args.dataset == 'deeplense':
        print(f"Loading DeepLense {args.deeplense_model} Dataset...")
        train_ds = DeepLenseDataset(root_dir='data', model=args.deeplense_model, split='train', physics_downsampler=physics)
        val_ds = DeepLenseDataset(root_dir='data', model=args.deeplense_model, split='val', physics_downsampler=physics)
        print(f"Loaded {len(train_ds)} train and {len(val_ds)} val samples.")
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    
    # 2. Model
    model = SimpleSRNet(in_channels=1).to(device)
    
    # 3. Trainer (Unsupervised Mode)
    results_dir = 'results_unsupervised'
    os.makedirs(results_dir, exist_ok=True)
    
    trainer = Trainer(model, train_loader, val_loader, device=device, lr=args.lr, 
                      results_dir=results_dir, physics=physics, 
                      mode='unsupervised', lambda_tv=args.lambda_tv)
    
    # 4. Train
    print(f"Starting PURE UNSUPERVISED Training (Lambda Phy: {args.lambda_phy}, Lambda TV: {args.lambda_tv})...")
    
    best_psnr = 0.0
    for epoch in range(1, args.epochs + 1):
        # The Trainer returns only running_total_loss right now, but internally logs phy/tv
        # We can still call train_epoch and log the average
        train_loss = trainer.train_epoch(epoch, lambda_physics=args.lambda_phy)
        
        # We evaluate on val using HR (even though it's unsupervised during train, testing requires HR for metrics)
        val_psnr, val_ssim = trainer.validate(epoch)
        
        print(f"Epoch {epoch}: Train Loss: {train_loss:.5f} | Val PSNR: {val_psnr:.2f} | Val SSIM: {val_ssim:.4f}")
        
        if val_psnr > best_psnr:
            best_psnr = val_psnr
            torch.save(trainer.model.state_dict(), f"{results_dir}/best_model.pth")
            print(f"Saved Best Unsupervised Model (PSNR: {best_psnr:.2f})")
        
        with open(f"{results_dir}/training_log.txt", "a") as f:
            f.write(f"{epoch},{train_loss},{val_psnr},{val_ssim}\n")
            
    print("Unsupervised Training Complete. Check results_unsupervised/ directory for logs and visual grids.")

if __name__ == "__main__":
    main()
