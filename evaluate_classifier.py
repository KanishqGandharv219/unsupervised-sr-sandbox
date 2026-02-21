import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score, roc_auc_score
from torch.utils.data import DataLoader

from src.data.classification_dataset import DeepLenseClassificationDataset
from src.models.classifier import SubstructureClassifier
import argparse

def evaluate_and_compare(data_dir='data/Model_I', out_prefix=''):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Evaluating models on data: {data_dir}...")
    
    # Modes to evaluate
    modes = ['bicubic', 'sr_hybrid', 'hr']
    classes = ['no_sub', 'vortex', 'subhalo']
    
    val_datasets = {
        'bicubic': DeepLenseClassificationDataset(data_dir, mode='bicubic', split='val'),
        'sr_hybrid': DeepLenseClassificationDataset(data_dir, mode='sr_hybrid', sr_model_path='results_hybrid/best_model.pth', split='val'),
        'hr': DeepLenseClassificationDataset(data_dir, mode='hr', split='val')
    }
    
    def get_predictions(mode, dataset):
        model = SubstructureClassifier(num_classes=3).to(device)
        model.load_state_dict(torch.load(f'results_classifier/model_{out_prefix}{mode}_best.pth', map_location=device))
        model.eval()
        
        loader = DataLoader(dataset, batch_size=32, shuffle=False)
        all_probs, all_labels = [], []
        
        with torch.no_grad():
            for imgs, labels in loader:
                imgs = imgs.to(device)
                outputs = model(imgs)
                probs = torch.softmax(outputs, dim=1)
                all_probs.append(probs.cpu().numpy())
                all_labels.append(labels.numpy())
                
        return np.vstack(all_probs), np.concatenate(all_labels)

    results = {}
    for mode in modes:
        probs, labels = get_predictions(mode, val_datasets[mode])
        preds = probs.argmax(axis=1)
        
        acc = accuracy_score(labels, preds)
        roc_auc = roc_auc_score(labels, probs, multi_class='ovr')
        
        results[mode] = {
            'probs': probs,
            'labels': labels,
            'preds': preds,
            'acc': acc,
            'auc': roc_auc
        }

    # Print Summary Table
    print("\n=== Classification Results ===")
    print(f"{'Input Type':<20} | {'Validation Accuracy':<20} | {'ROC-AUC (OvR)':<20}")
    print("-" * 68)
    for mode in modes:
        print(f"{mode:<20} | {results[mode]['acc']:<20.4f} | {results[mode]['auc']:<20.4f}")
    
    # 1. Plot ROC Curves
    plt.style.use('default')
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors = {'bicubic': 'blue', 'sr_hybrid': 'green', 'hr': 'red'}
    linestyles = {'bicubic': '--', 'sr_hybrid': '-', 'hr': ':'}
    
    for i, class_name in enumerate(classes):
        ax = axes[i]
        for mode in modes:
            probs = results[mode]['probs']
            labels = results[mode]['labels']
            
            fpr, tpr, _ = roc_curve((labels == i).astype(int), probs[:, i])
            class_auc = auc(fpr, tpr)
            
            label_name = 'SR Hybrid (Ours)' if mode == 'sr_hybrid' else mode.capitalize()
            ax.plot(fpr, tpr, color=colors[mode], linestyle=linestyles[mode],
                    label=f'{label_name} (AUC={class_auc:.3f})', linewidth=2)
            
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'ROC Curve: {class_name}')
        ax.legend(loc="lower right")
        ax.grid(alpha=0.3)
        
    plt.tight_layout()
    plt.savefig(f'results_classifier/{out_prefix}roc_comparison.png', dpi=150)
    plt.close()
    
    # 2. Confusion Matrices
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    cm_cmaps = {'bicubic': 'Blues', 'sr_hybrid': 'Greens', 'hr': 'Reds'}
    
    for i, mode in enumerate(modes):
        cm = confusion_matrix(results[mode]['labels'], results[mode]['preds'])
        sns.heatmap(cm, annot=True, fmt='d', cmap=cm_cmaps[mode], 
                    xticklabels=classes, yticklabels=classes, ax=axes[i], cbar=False)
        
        title = 'SR Hybrid (Ours)' if mode == 'sr_hybrid' else mode.capitalize()
        axes[i].set_title(f'Confusion Matrix: {title}')
        axes[i].set_ylabel('True Label')
        axes[i].set_xlabel('Predicted Label')
        
    plt.tight_layout()
    plt.savefig(f'results_classifier/{out_prefix}confusion_matrices.png', dpi=150)
    plt.close()
    
    print(f"\nVisualizations saved to results_classifier/ with prefix '{out_prefix}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/Model_I")
    parser.add_argument("--out_prefix", type=str, default="")
    args = parser.parse_args()
    
    evaluate_and_compare(data_dir=args.data_dir, out_prefix=args.out_prefix)
