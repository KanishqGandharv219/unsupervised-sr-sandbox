import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from scipy.stats import wilcoxon
import torch.nn.functional as F

from src.data.deeplense_dataset import DeepLenseDataset
from models.baseline import SimpleSRNet
from utils.physics import PhysicsDownsampler
from src.metrics.lens_analysis import arc_sharpness_score, ring_contrast_score

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Load Dataset
    print("Loading Validation Dataset...")
    physics = PhysicsDownsampler(scale_factor=2, device=device)
    
    # Let's use the Model_I dataset we created earlier
    val_dataset = DeepLenseDataset(root_dir='data', model='Model_I', split='val', physics_downsampler=physics)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    
    print(f"Validation Samples: {len(val_dataset)}")
    
    # 2. Load Models
    # Initialize models
    baseline_model = SimpleSRNet(in_channels=1).to(device)
    hybrid_model = SimpleSRNet(in_channels=1).to(device)
    unsupervised_model = SimpleSRNet(in_channels=1).to(device)
    
    # Check if weights exist (skip if missing so we don't crash)
    weights = {
        'sr_baseline': 'results/best_model.pth',
        'sr_hybrid': 'results_hybrid/best_model.pth',
        'sr_unsupervised': 'results_unsupervised/best_model.pth'
    }
    
    models = {}
    for name, path in weights.items():
        if os.path.exists(path):
            if name == 'sr_baseline':
                baseline_model.load_state_dict(torch.load(path, map_location=device))
                baseline_model.eval()
                models[name] = baseline_model
            elif name == 'sr_hybrid':
                hybrid_model.load_state_dict(torch.load(path, map_location=device))
                hybrid_model.eval()
                models[name] = hybrid_model
            elif name == 'sr_unsupervised':
                unsupervised_model.load_state_dict(torch.load(path, map_location=device))
                unsupervised_model.eval()
                models[name] = unsupervised_model
            print(f"Loaded {name} from {path}")
        else:
            print(f"Warning: Model weights not found at {path}. Will skip this model in analysis if not running later.")
            models[name] = None
        
    # 3. Collect Metrics
    results = {
        'bicubic': {'sharpness': [], 'contrast': []},
        'hr': {'sharpness': [], 'contrast': []}
    }
    for name in models.keys():
        if models[name] is not None:
            results[name] = {'sharpness': [], 'contrast': []}
            
    print("Computing Scienctific Metrics on Validation Set...")
    with torch.no_grad():
        for lr_imgs, hr_imgs, labels in val_loader:
            lr_imgs = lr_imgs.to(device)
            hr_imgs = hr_imgs.to(device)
            
            # Upsample
            bicubic = F.interpolate(lr_imgs, size=hr_imgs.shape[-2:], mode='bicubic', align_corners=False).clamp(0, 1)
            
            # Predict
            sr_preds = {}
            for name, model in models.items():
                if model is not None:
                    sr_preds[name] = model(lr_imgs).clamp(0, 1)
                    
            # Convert to numpy for metric calculation (batch_size=1)
            lr_np = lr_imgs[0, 0].cpu().numpy()
            hr_np = hr_imgs[0, 0].cpu().numpy()
            bicubic_np = bicubic[0, 0].cpu().numpy()
            
            # Compute baseline metrics
            
            results['hr']['sharpness'].append(arc_sharpness_score(hr_np))
            results['hr']['contrast'].append(ring_contrast_score(hr_np))
            
            results['bicubic']['sharpness'].append(arc_sharpness_score(bicubic_np))
            results['bicubic']['contrast'].append(ring_contrast_score(bicubic_np))
            
            # Compute Neural Network Metrics
            for name, pred in sr_preds.items():
                pred_np = pred[0, 0].cpu().numpy()
                results[name]['sharpness'].append(arc_sharpness_score(pred_np))
                results[name]['contrast'].append(ring_contrast_score(pred_np))
                
    # 4. Statistical Summary
    print("\n" + "="*50)
    print("Arc Sharpness Score (mean +/- std):")
    for method in results.keys():
        scores = results[method]['sharpness']
        if len(scores) > 0:
            print(f"{method:15s}: {np.mean(scores):.3f} +/- {np.std(scores):.3f}")
            
    print("\nRing Contrast Ratio (mean +/- std):")
    for method in results.keys():
        scores = results[method]['contrast']
        if len(scores) > 0:
            print(f"{method:15s}: {np.mean(scores):.3f} +/- {np.std(scores):.3f}")
            
    # 5. Wilcoxon Signed-Rank Test
    if 'sr_hybrid' in results and 'bicubic' in results:
        hybrid_scores = results['sr_hybrid']['sharpness']
        bicubic_scores = results['bicubic']['sharpness']
        
        if len(hybrid_scores) > 0 and len(bicubic_scores) > 0:
            try:
                stat, p_value = wilcoxon(bicubic_scores, hybrid_scores)
                print("\n" + "="*50)
                print("Wilcoxon Signed-Rank Test (Bicubic vs SR Hybrid on Arc Sharpness)")
                print(f"Statistic: {stat}")
                print(f"P-Value: {p_value:.3e}")
                if p_value < 0.05:
                    print("Result: STATISTICALLY SIGNIFICANT IMPROVEMENT (p < 0.05)")
                else:
                    print("Result: NOT statistically significant")
            except Exception as e:
                print(f"Could not compute Wilcoxon test: {e}")
                
    if 'sr_hybrid' in results and 'sr_baseline' in results:
        hybrid_scores = results['sr_hybrid']['sharpness']
        baseline_scores = results['sr_baseline']['sharpness']
        
        if len(hybrid_scores) > 0 and len(baseline_scores) > 0:
            try:
                stat, p_value = wilcoxon(baseline_scores, hybrid_scores)
                print("\n" + "-"*50)
                print("Wilcoxon Signed-Rank Test (Baseline vs SR Hybrid on Arc Sharpness)")
                print(f"Statistic: {stat}")
                print(f"P-Value: {p_value:.3e}")
                if p_value < 0.05:
                    print("Result: STATISTICALLY SIGNIFICANT IMPROVEMENT (p < 0.05)")
                else:
                    print("Result: NOT statistically significant")
            except Exception as e:
                print(f"Could not compute Wilcoxon test: {e}")
                
    # 6. Plotting
    print("\nGenerating visual reports...")
    os.makedirs('results', exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    valid_methods = [m for m in ['lr', 'bicubic', 'sr_baseline', 'sr_hybrid', 'sr_unsupervised', 'hr'] if m in results and len(results[m]['sharpness']) > 0]
    labels = [m.replace('sr_', '').capitalize() for m in valid_methods]
    
    # Sharpness
    sharpness_data = [results[m]['sharpness'] for m in valid_methods]
    axes[0].boxplot(sharpness_data, tick_labels=labels)
    axes[0].set_ylabel('Arc Sharpness Score')
    axes[0].set_title('Einstein Ring Edge Sharpness')
    axes[0].grid(alpha=0.3)
    
    # Contrast
    contrast_data = [results[m]['contrast'] for m in valid_methods]
    axes[1].boxplot(contrast_data, tick_labels=labels)
    axes[1].set_ylabel('Ring Contrast Ratio')
    axes[1].set_title('Ring vs Background Contrast')
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/lens_analysis_metrics.png', dpi=150)
    print("Saved plots to results/lens_analysis_metrics.png")

if __name__ == '__main__':
    main()
