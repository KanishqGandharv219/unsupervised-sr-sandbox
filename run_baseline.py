
import argparse
import torch
from torch.utils.data import DataLoader, random_split
from data.lensing_dataset import SyntheticLensingDataset
from models.baseline import SimpleSRNet
from training.trainer import Trainer

def main():
    parser = argparse.ArgumentParser(description="DeepLense-SR Baseline Training")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--samples", type=int, default=1000, help="Number of synthetic samples")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Dataset & Dataloaders
    print("Generating Synthetic Lensing Data...")
    dataset = SyntheticLensingDataset(num_samples=args.samples)
    
    # Split 80/20
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    
    # 2. Model
    model = SimpleSRNet(in_channels=1).to(device)
    
    # 3. Trainer
    trainer = Trainer(model, train_loader, val_loader, device=device, lr=args.lr)
    
    # 4. Train
    print("Starting Training...")
    trainer.fit(epochs=args.epochs)
    print("Training Complete. Check results/ directory.")

if __name__ == "__main__":
    main()
