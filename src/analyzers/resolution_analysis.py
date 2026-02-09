"""
Resolution Inconsistency Detection

Detects when regions of a document have different resolutions.
Common in fraud when high-res text/images are pasted onto low-res scans.
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple

def analyze_resolution_inconsistency(image_path: str, output_path: str = None) -> Dict:
    """
    Detect resolution inconsistencies across a document image.
    
    Strategy:
    1. Divide image into grid patches
    2. For each patch, measure "local sharpness" via edge strength
    3. Flag patches with significantly different sharpness
    
    High sharpness = high resolution / sharp edges
    Low sharpness = low resolution / blurry edges
    
    Args:
        image_path: Path to document image
        output_path: Optional path to save heatmap
        
    Returns:
        Dictionary with analysis results
    """
    try:
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            return {"error": "Could not read image"}
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Compute sharpness map
        sharpness_map, patch_size = _compute_sharpness_map(gray)
        
        # Analyze sharpness distribution
        sharpness_values = sharpness_map[sharpness_map > 0]  # Exclude empty patches
        
        if len(sharpness_values) < 4:
            return {
                "is_suspicious": False,
                "num_patches": len(sharpness_values),
                "interpretation": "Insufficient patches for analysis"
            }
        
        # Statistical analysis
        mean_sharpness = np.mean(sharpness_values)
        std_sharpness = np.std(sharpness_values)
        cv_sharpness = std_sharpness / (mean_sharpness + 1e-5)
        
        # Find outlier patches (using IQR method)
        q1 = np.percentile(sharpness_values, 25)
        q3 = np.percentile(sharpness_values, 75)
        iqr = q3 - q1
        
        # Outliers are values outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR]
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        outlier_mask = (sharpness_map < lower_bound) | (sharpness_map > upper_bound)
        num_outliers = np.sum(outlier_mask)
        total_patches = np.sum(sharpness_map > 0)
        outlier_ratio = num_outliers / total_patches if total_patches > 0 else 0
        
        # Threshold: CV > 0.4 OR outlier_ratio > 0.15 is suspicious
        is_suspicious = (cv_sharpness > 0.4) or (outlier_ratio > 0.15)
        
        results = {
            "is_suspicious": bool(is_suspicious),
            "cv_sharpness": float(cv_sharpness),
            "outlier_ratio": float(outlier_ratio),
            "num_outliers": int(num_outliers),
            "total_patches": int(total_patches),
            "mean_sharpness": float(mean_sharpness),
            "std_sharpness": float(std_sharpness),
            "interpretation": "High CV or outlier ratio suggests pasted high-res content on low-res scan"
        }
        
        # Visualization
        if output_path:
            _visualize_sharpness_map(img, sharpness_map, outlier_mask, patch_size, output_path)
        
        return results
        
    except Exception as e:
        return {"error": str(e)}


def _compute_sharpness_map(gray: np.ndarray, patch_size: int = 64) -> Tuple[np.ndarray, int]:
    """
    Compute local sharpness for each patch.
    
    Sharpness is measured using Laplacian variance (common metric).
    
    Returns:
        (sharpness_map, patch_size) where sharpness_map has reduced dimensions
    """
    h, w = gray.shape
    
    # Compute Laplacian (edge detector)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    
    # Divide into patches and compute variance
    rows = h // patch_size
    cols = w // patch_size
    
    sharpness_map = np.zeros((rows, cols))
    
    for i in range(rows):
        for j in range(cols):
            y1, y2 = i * patch_size, (i + 1) * patch_size
            x1, x2 = j * patch_size, (j + 1) * patch_size
            
            patch_laplacian = laplacian[y1:y2, x1:x2]
            
            # Sharpness = variance of Laplacian
            # High variance = sharp edges, low variance = blurry
            sharpness = np.var(patch_laplacian)
            sharpness_map[i, j] = sharpness
    
    return sharpness_map, patch_size


def _visualize_sharpness_map(img: np.ndarray, sharpness_map: np.ndarray, 
                              outlier_mask: np.ndarray, patch_size: int, 
                              output_path: str):
    """Generate heatmap visualization of sharpness."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    # Upsample sharpness map to original image size
    h, w = img.shape[:2]
    sharpness_upsampled = cv2.resize(sharpness_map, (w, h), interpolation=cv2.INTER_NEAREST)
    outlier_upsampled = cv2.resize(outlier_mask.astype(np.uint8), (w, h), 
                                    interpolation=cv2.INTER_NEAREST)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Original image
    axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Original Document', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # Sharpness heatmap
    im = axes[1].imshow(sharpness_upsampled, cmap='jet', alpha=0.7)
    axes[1].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cmap='gray', alpha=0.3)
    axes[1].set_title('Sharpness Heatmap\n(Red=Sharp/High-Res, Blue=Blurry/Low-Res)', 
                      fontsize=14, fontweight='bold')
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    
    # Outlier overlay
    outlier_overlay = img.copy()
    outlier_overlay[outlier_upsampled > 0] = [0, 0, 255]  # Red for outliers
    
    # Blend
    blended = cv2.addWeighted(img, 0.6, outlier_overlay, 0.4, 0)
    axes[2].imshow(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB))
    axes[2].set_title('Outlier Regions (Red)', fontsize=14, fontweight='bold')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Resolution analysis visualization saved to {output_path}")
