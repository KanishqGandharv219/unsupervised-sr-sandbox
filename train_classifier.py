import os
import argparse
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
import numpy as np

from src.data.classification_dataset import DeepLenseClassificationDataset
from src.models.classifier import SubstructureClassifier

def train_classifier(mode, data_dir='data/Model_I', out_prefix='', sr_model_path=None, epochs=30, lr=0.001, batch_size=32):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device} | Mode: {mode} | Data: {data_dir}")
    
    # Data loaders
    train_dataset = DeepLenseClassificationDataset(data_dir, mode=mode, 
                                                   sr_model_path=sr_model_path, split='train')
    val_dataset = DeepLenseClassificationDataset(data_dir, mode=mode,
                                                 sr_model_path=sr_model_path, split='val')
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Model, loss, optimizer
    model = SubstructureClassifier(num_classes=3).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_val_auc = 0
    results = {'train_loss': [], 'val_acc': [], 'val_auc': []}
    
    os.makedirs('results_classifier', exist_ok=True)
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        scheduler.step()
        
        # Validation
        model.eval()
        all_preds, all_labels, all_probs = [], [], []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device)
                outputs = model(imgs)
                probs = torch.softmax(outputs, dim=1)
                preds = outputs.argmax(dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())
                all_probs.extend(probs.cpu().numpy())
        
        # Metrics
        val_acc = (np.array(all_preds) == np.array(all_labels)).mean()
        val_auc = roc_auc_score(all_labels, all_probs, multi_class='ovr')
        
        avg_train_loss = train_loss / len(train_loader)
        results['train_loss'].append(avg_train_loss)
        results['val_acc'].append(float(val_acc))
        results['val_auc'].append(float(val_auc))
        
        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_train_loss:.4f} | "
              f"Val Acc: {val_acc:.4f} | Val AUC: {val_auc:.4f}")
        
        # Save best model based on AUC
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), f'results_classifier/model_{out_prefix}{mode}_best.pth')
            print(f" -> Best model saved with AUC {best_val_auc:.4f}")
    
    # Save results
    with open(f'results_classifier/{out_prefix}{mode}_metrics.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="bicubic", choices=["bicubic", "sr_hybrid", "hr"])
    parser.add_argument("--data_dir", type=str, default="data/Model_I")
    parser.add_argument("--out_prefix", type=str, default="")
    parser.add_argument("--sr_model", type=str, default="results_hybrid/best_model.pth")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()
    
    train_classifier(args.mode, data_dir=args.data_dir, out_prefix=args.out_prefix, sr_model_path=args.sr_model, epochs=args.epochs, lr=args.lr, batch_size=args.batch_size)
