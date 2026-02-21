
import torch
import torch.nn.functional as F
import numpy as np

def gaussian_kernel(kernel_size=5, sigma=1.0, channels=1):
    # Create a x, y coordinate grid of shape (kernel_size, kernel_size)
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

    mean = (kernel_size - 1) / 2.
    variance = sigma**2.

    # Calculate the 2-D gaussian kernel
    gaussian_kernel = (1. / (2. * np.pi * variance)) * \
                      torch.exp(
                          -torch.sum((xy_grid - mean)**2., dim=-1) / \
                          (2 * variance)
                      )

    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    # Reshape to 2d depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

    return gaussian_kernel

class PhysicsDownsampler:
    """
    Simulates the forward degradation physics:
    HR Image -> Convolve (PSF) -> Decimate (Pixelate) -> Add Noise -> LR Image
    """
    def __init__(self, scale_factor=2, psf_sigma=1.0, noise_level=0.01, device='cpu'):
        self.scale_factor = scale_factor
        self.psf_sigma = psf_sigma
        self.noise_level = noise_level
        self.device = device
        
        # Precompute the PSF kernel
        self.kernel = gaussian_kernel(kernel_size=5, sigma=psf_sigma, channels=1).to(device)

    def __call__(self, hr_img, deterministic=False):
        """
        Args:
            hr_img: Tensor of shape (1, H, W) or (B, 1, H, W) in range [0, 1]
            deterministic: If True, do not add noise. Useful for loss calculation.
        Returns:
            lr_img: Tensor of shape (..., H/s, W/s)
        """
        # Ensure 4D
        if hr_img.dim() == 3:
            hr_img = hr_img.unsqueeze(0)
        
        # 1. Convolve with PSF (Blur)
        # Pad to keep size same before decimation
        pad = 2 # kernel_size // 2
        blurred = F.conv2d(hr_img, self.kernel, padding=pad, groups=1)
        
        # 2. Decimate (Average Pooling or simple subsampling)
        # Here we use Area interpolation which mimics sensor integration
        lr_img = F.interpolate(blurred, scale_factor=1/self.scale_factor, mode='area')
        
        # 3. Add Noise
        if not deterministic:
            noise = torch.randn_like(lr_img) * self.noise_level
            lr_img = lr_img + noise
        
        return lr_img
