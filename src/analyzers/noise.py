"""
Noise Variance Analysis Module - Enhanced for Documents

Detects manipulation by analyzing noise distribution patterns.
Spliced or edited regions often have inconsistent noise compared to original content.

ENHANCED: Now includes regional analysis and quantitative metrics.
"""

import cv2
import numpy as np
from typing import Dict

def analyze_noise(image_path: str, output_path: str = None) -> Dict:
    """
    Performs enhanced noise variance analysis with regional metrics.
    
    Args:
        image_path: Path to the source image
        output_path: Optional path to save visualization
        
    Returns:
        Dictionary with noise metrics and verdict
    """
    try:
        image = cv2.imread(image_path)
        if image is None:
            return {"error": "Could not read image"}
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Extract noise residual
        denoised = cv2.medianBlur(gray, 5)
        noise_map = cv2.absdiff(gray, denoised)
        
        # Regional analysis: divide into grid
        patch_size = 64
        h, w = gray.shape
        rows = h // patch_size
        cols = w // patch_size
        
        patch_variances = []
        patch_means = []
        
        for i in range(rows):
            for j in range(cols):
                y1, y2 = i * patch_size, (i + 1) * patch_size
                x1, x2 = j * patch_size, (j + 1) * patch_size
                
                patch = noise_map[y1:y2, x1:x2]
                patch_variances.append(np.var(patch))
                patch_means.append(np.mean(patch))
        
        if len(patch_variances) < 4:
            return {
                "is_suspicious": False,
                "interpretation": "Image too small for patch analysis"
            }
        
        # Compute statistics
        global_variance = np.var(noise_map)
        global_mean = np.mean(noise_map)
        
        patch_variances = np.array(patch_variances)
        patch_means = np.array(patch_means)
        
        # Coefficient of variation across patches
        variance_cv = np.std(patch_variances) / (np.mean(patch_variances) + 1e-5)
        mean_cv = np.std(patch_means) / (np.mean(patch_means) + 1e-5)
        
        # Detect outlier patches (inconsistent noise = tampering)
        q1 = np.percentile(patch_variances, 25)
        q3 = np.percentile(patch_variances, 75)
        iqr = q3 - q1
        
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        outlier_mask = (patch_variances < lower_bound) | (patch_variances > upper_bound)
        num_outliers = np.sum(outlier_mask)
        outlier_ratio = num_outliers / len(patch_variances)
        
        # Verdict
        # High CV or many outliers suggests tampering
        is_suspicious = (variance_cv > 0.5) or (outlier_ratio > 0.15)
        
        results = {
            "global_variance": float(global_variance),
            "global_mean": float(global_mean),
            "variance_cv": float(variance_cv),
            "mean_cv": float(mean_cv),
            "num_patches": len(patch_variances),
            "num_outliers": int(num_outliers),
            "outlier_ratio": float(outlier_ratio),
            "is_suspicious": bool(is_suspicious),
            "interpretation": "High CV (>0.5) or outlier ratio (>15%) indicates inconsistent noise (tampering)"
        }
        
        # Visualization
        if output_path:
            _visualize_noise(image, noise_map, patch_variances, rows, cols, 
                           patch_size, outlier_mask, output_path)
        
        return results
        
    except Exception as e:
        return {"error": str(e)}


def _visualize_noise(image: np.ndarray, noise_map: np.ndarray, 
                     patch_variances: np.ndarray, rows: int, cols: int, 
                     patch_size: int, outlier_mask: np.ndarray, output_path: str):
    """Generate noise analysis visualization with heatmap."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    # Create variance heatmap
    variance_grid = patch_variances.reshape(rows, cols)
    
    # Upsample to original size
    h, w = image.shape[:2]
    variance_upsampled = cv2.resize(variance_grid, (w, h), interpolation=cv2.INTER_NEAREST)
    
    # Outlier overlay
    outlier_grid = outlier_mask.reshape(rows, cols).astype(np.uint8)
    outlier_upsampled = cv2.resize(outlier_grid, (w, h), interpolation=cv2.INTER_NEAREST)
    
    overlay = image.copy()
    overlay[outlier_upsampled > 0] = [0, 0, 255]  # Red for outliers
    blended = cv2.addWeighted(image, 0.6, overlay, 0.4, 0)
    
    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    axes[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('Original Image', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    # Noise map with colormap
    im1 = axes[0, 1].imshow(noise_map, cmap='hot')
    axes[0, 1].set_title('Noise Map\n(Bright = High Noise)', fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)
    
    # Variance heatmap
    im2 = axes[1, 0].imshow(variance_upsampled, cmap='jet', alpha=0.7)
    axes[1, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cmap='gray', alpha=0.3)
    axes[1, 0].set_title('Noise Variance Heatmap\n(Red = High, Blue = Low)', fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')
    plt.colorbar(im2, ax=axes[1, 0], fraction=0.046, pad=0.04)
    
    # Outlier overlay
    axes[1, 1].imshow(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB))
    axes[1, 1].set_title(f'Outlier Regions (Red)\n({np.sum(outlier_mask)} patches)', 
                         fontsize=12, fontweight='bold')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Noise analysis visualization saved to {output_path}")
