import torch
import torch.nn as nn
import torchvision.models as models

class SubstructureClassifier(nn.Module):
    def __init__(self, num_classes=3, pretrained=False):
        super().__init__()
        # Use ResNet18 backbone
        self.backbone = models.resnet18(pretrained=pretrained)
        
        # Modify first conv layer for single-channel input
        self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Replace final FC layer for 3 classes
        self.backbone.fc = nn.Linear(512, num_classes)
    
    def forward(self, x):
        return self.backbone(x)

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 32x32
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 16x16
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), # 8x8
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1) # 1x1
        )
        self.classifier = nn.Linear(256, num_classes)
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

if __name__ == "__main__":
    # Test forward pass
    x = torch.randn(4, 1, 64, 64)
    model1 = SubstructureClassifier()
    model2 = SimpleCNN()
    
    out1 = model1(x)
    out2 = model2(x)
    
    print("ResNet18 output shape:", out1.shape)
    print("SimpleCNN output shape:", out2.shape)
