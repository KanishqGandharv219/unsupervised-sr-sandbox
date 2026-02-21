import numpy as np
import cv2

def arc_sharpness_score(img):
    """
    Quantify Einstein ring/arc edge sharpness using gradient magnitude.
    Higher score = sharper, better-defined arcs.
    
    Args:
        img: 2D numpy array, normalized [0, 1]
    """
    img_uint8 = (img * 255).astype(np.uint8)
    
    sobel_x = cv2.Sobel(img_uint8, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(img_uint8, cv2.CV_64F, 0, 1, ksize=3)
    gradient_mag = np.sqrt(sobel_x**2 + sobel_y**2)
    
    brightness_threshold = np.percentile(img, 75)
    arc_mask = img > brightness_threshold
    
    if arc_mask.sum() > 0:
        sharpness = gradient_mag[arc_mask].mean()
    else:
        sharpness = 0.0
    
    return sharpness

def ring_contrast_score(img, inner_r=8, outer_r=28):
    """
    Measure contrast between Einstein ring annulus and background.
    """
    h, w = img.shape
    center_y, center_x = h // 2, w // 2
    
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((Y - center_y)**2 + (X - center_x)**2)
    
    ring_mask = (dist >= inner_r) & (dist <= outer_r)
    background_mask = (dist < inner_r) | (dist > outer_r)
    
    ring_brightness = img[ring_mask].mean() if ring_mask.sum() > 0 else 0
    background_brightness = img[background_mask].mean() if background_mask.sum() > 0 else 1e-8
    
    contrast = ring_brightness / (background_brightness + 1e-8)
    return contrast
