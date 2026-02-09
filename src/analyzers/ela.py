"""
Error Level Analysis (ELA) Module - Enhanced for Documents

Detects manipulation by re-saving an image at a specific quality and comparing it to the original.
Areas that have been edited will have different compression artifacts.

ENHANCED: Now includes quantitative metrics and suspicious region detection.
"""

import cv2
import numpy as np
from PIL import Image, ImageChops
import os
from typing import Dict

def perform_ela(image_path: str, quality: int = 90, output_path: str = None) -> Dict:
    """
    Performs Error Level Analysis with quantitative metrics.
    
    **IMPORTANT**: ELA only works on JPEG images. PNG/lossless formats will be skipped.
    
    Args:
        image_path: Path to the source image
        quality: JPEG quality for re-compression (default: 90)
        output_path: Optional path to save visualization
        
    Returns:
        Dictionary with ELA metrics and verdict
    """
    try:
        # Load image and check format
        original = Image.open(image_path)
        
        # ELA ONLY works on JPEG images (relies on compression artifacts)
        if original.format not in ['JPEG', 'JPG']:
            return {
                "is_suspicious": False,
                "skipped": True,
                "reason": f"ELA only works on JPEG images (got {original.format})",
                "interpretation": "PNG/lossless formats cannot be analyzed with ELA"
            }
        
        original = original.convert('RGB')
        
        # Save and reload at specified quality
        temp_filename = 'temp_ela.jpg'
        original.save(temp_filename, 'JPEG', quality=quality)
        resaved = Image.open(temp_filename).convert('RGB')
        
        # Calculate pixel-level difference
        ela_image = ImageChops.difference(original, resaved)
        
        # Convert to numpy for analysis
        ela_np = np.array(ela_image)
        
        # Compute metrics
        mean_error = np.mean(ela_np)
        max_error = np.max(ela_np)
        std_error = np.std(ela_np)
        
        # Detect suspicious regions
        # High-error regions indicate different compression history (edited areas)
        threshold = mean_error + 2 * std_error
        suspicious_mask = np.any(ela_np > threshold, axis=2)  # Any channel exceeds threshold
        
        suspicious_pixels = np.sum(suspicious_mask)
        total_pixels = suspicious_mask.size
        suspicious_ratio = suspicious_pixels / total_pixels
        
        # Segment into regions
        suspicious_mask_uint8 = suspicious_mask.astype(np.uint8) * 255
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(suspicious_mask_uint8)
        
        # Filter small regions (noise)
        min_region_size = total_pixels * 0.005  # At least 0.5% of image
        significant_regions = []
        
        for i in range(1, num_labels):  # Skip background (label 0)
            area = stats[i, cv2.CC_STAT_AREA]
            if area > min_region_size:
                x = stats[i, cv2.CC_STAT_LEFT]
                y = stats[i, cv2.CC_STAT_TOP]
                w = stats[i, cv2.CC_STAT_WIDTH]
                h = stats[i, cv2.CC_STAT_HEIGHT]
                significant_regions.append({
                    'bbox': (x, y, w, h),
                    'area': int(area),
                    'area_ratio': float(area / total_pixels)
                })
        
        # Verdict
        # Documents: suspicious if >5% of pixels have high error OR multiple large regions
        is_suspicious = (suspicious_ratio > 0.05) or (len(significant_regions) > 2)
        
        results = {
            "mean_error": float(mean_error),
            "max_error": float(max_error),
            "std_error": float(std_error),
            "suspicious_ratio": float(suspicious_ratio),
            "num_suspicious_regions": len(significant_regions),
            "suspicious_regions": significant_regions[:10],  # Top 10
            "is_suspicious": bool(is_suspicious),
            "interpretation": "High error ratio (>5%) or multiple regions suggests pasted/edited content"
        }
        
        # Visualization
        if output_path:
            _visualize_ela(original, ela_np, suspicious_mask, significant_regions, output_path)
        
        # Cleanup
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
        
        return results
        
    except Exception as e:
        return {"error": str(e)}


def _visualize_ela(original: Image.Image, ela_np: np.ndarray, 
                   suspicious_mask: np.ndarray, regions: list, output_path: str):
    """Generate ELA visualization with highlighted suspicious regions."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    # Convert original to numpy
    orig_np = np.array(original)
    
    # Create enhanced ELA (scale for visibility)
    ela_enhanced = np.clip(ela_np * 10, 0, 255).astype(np.uint8)
    
    # Create overlay
    overlay = orig_np.copy()
    overlay[suspicious_mask] = [255, 0, 0]  # Red for suspicious
    blended = cv2.addWeighted(orig_np, 0.6, overlay, 0.4, 0)
    
    # Draw bounding boxes
    for region in regions[:10]:  # Top 10
        x, y, w, h = region['bbox']
        cv2.rectangle(blended, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    axes[0].imshow(orig_np)
    axes[0].set_title('Original Image', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    axes[1].imshow(ela_enhanced)
    axes[1].set_title('ELA (Enhanced 10x)\n(Bright = High Error)', fontsize=12, fontweight='bold')
    axes[1].axis('off')
    
    axes[2].imshow(blended)
    axes[2].set_title(f'Suspicious Regions ({len(regions)})\n(Red = High Error, Green Box = Region)', 
                      fontsize=12, fontweight='bold')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"ELA visualization saved to {output_path}")
