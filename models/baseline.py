
import torch
import torch.nn as nn

class SimpleSRNet(nn.Module):
    """
    A simple SR network for single-channel lensing data.
    Uses a residual-like structure with PixelShuffle for upsampling.
    Scale Factor is hardcoded to 2 for now.
    """
    def __init__(self, in_channels=1, features=32):
        super(SimpleSRNet, self).__init__()
        
        # Encoder: Extract features from LR
        self.conv1 = nn.Conv2d(in_channels, features, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        
        self.res_block1 = nn.Sequential(
            nn.Conv2d(features, features, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, features, 3, padding=1)
        )
        
        self.res_block2 = nn.Sequential(
            nn.Conv2d(features, features, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, features, 3, padding=1)
        )
        
        # Upsampler: Upscale features -> HR
        self.upsample = nn.Sequential(
            nn.Conv2d(features, features * 4, 3, padding=1), # For 2x scale
            nn.PixelShuffle(2),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, in_channels, 3, padding=1)
        )

    def forward(self, x):
        # x: (B, 1, H, W)
        x1 = self.relu(self.conv1(x))
        
        # Residual Blocks
        r1 = self.res_block1(x1)
        x2 = x1 + r1
        
        r2 = self.res_block2(x2)
        x3 = x2 + r2
        
        # Upsampling
        out = self.upsample(x3)
        return out
