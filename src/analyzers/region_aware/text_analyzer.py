"""
Text Region Analyzer

Compares text regions to each other for consistency.
Only compares text-to-text (not text-to-photo), eliminating false positives.
"""

import cv2
import numpy as np
from typing import Dict, List

def analyze_text_regions(regions: List[Dict], image: np.ndarray) -> Dict:
    """
    Compare text regions for consistency.
    
    Args:
        regions: List of region dictionaries
        image: Original image (BGR)
        
    Returns:
        Analysis results with consistency metrics
    """
    from ..region_aware.segmenter import REGION_TEXT
    
    # Filter for text regions only
    text_regions = [r for r in regions if r['type'] == REGION_TEXT]
    
    if len(text_regions) < 2:
        return {
            'is_suspicious': False,
            'num_text_regions': len(text_regions),
            'reason': 'Insufficient text regions for comparison (need at least 2)'
        }
    
    # Extract features from each text region
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    features = []
    for region in text_regions:
        mask = region['mask']
        region_gray = gray.copy()
        region_gray[~mask] = 255  # Set non-region to white
        
        # Extract bounding box
        x, y, w, h = region['bbox']
        region_crop = region_gray[y:y+h, x:x+w]
        
        feature_set = _extract_text_features(region_crop)
        features.append(feature_set)
    
    # Compare features across regions
    stroke_widths = [f['stroke_width'] for f in features]
    char_heights = [f['char_height'] for f in features]
    sharpness_vals = [f['sharpness'] for f in features]
    noise_vars = [f['noise_variance'] for f in features]
    
    # Compute coefficient of variation (CV) across text regions
    stroke_cv = _coeff_variation(stroke_widths)
    char_cv = _coeff_variation(char_heights)
    sharpness_cv = _coeff_variation(sharpness_vals)
    noise_cv = _coeff_variation(noise_vars)
    
    # Decision logic
    # Text regions should be consistent (low CV)
    # Increased thresholds since we're only comparing text-to-text now
    is_suspicious = (
        (stroke_cv > 0.5) or      # Was 0.3, now 0.5 (more lenient)
        (char_cv > 0.5) or        # Was 0.3
        (sharpness_cv > 0.6) or   # Was 0.4
        (noise_cv > 0.8)          # Was 0.5
    )
    
    # Identify outlier regions
    outliers = []
    stroke_mean = np.mean(stroke_widths)
    stroke_std = np.std(stroke_widths)
    
    for i, region in enumerate(text_regions):
        z_score = abs(stroke_widths[i] - stroke_mean) / (stroke_std + 1e-5)
        if z_score > 2.5:  # More than 2.5 std deviations
            outliers.append({
                'region_id': region['id'],
                'bbox': region['bbox'],
                'z_score': float(z_score),
                'stroke_width': stroke_widths[i]
            })
    
    return {
        'is_suspicious': bool(is_suspicious),
        'num_text_regions': len(text_regions),
        'stroke_cv': float(stroke_cv),
        'char_cv': float(char_cv),
        'sharpness_cv': float(sharpness_cv),
        'noise_cv': float(noise_cv),
        'outliers': outliers,
        'interpretation': 'Comparing text regions only. High CV suggests pasted text from different sources.'
    }


def _extract_text_features(region_crop: np.ndarray) -> Dict:
    """
    Extract font/text features from a text region.
    """
    # Binarize
    _, binary = cv2.threshold(region_crop, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # 1. Stroke width (distance transform)
    dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    stroke_width = np.mean(dist_transform[dist_transform > 0]) * 2 if np.any(dist_transform > 0) else 0
    
    # 2. Character height (connected components)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    
    if num_labels > 1:
        heights = stats[1:, cv2.CC_STAT_HEIGHT]
        char_height = np.median(heights) if len(heights) > 0 else 0
    else:
        char_height = 0
    
    # 3. Sharpness (Laplacian variance)
    laplacian = cv2.Laplacian(region_crop, cv2.CV_64F)
    sharpness = np.var(laplacian)
    
    # 4. Noise variance
    denoised = cv2.medianBlur(region_crop, 3)
    noise = cv2.absdiff(region_crop, denoised)
    noise_variance = np.var(noise)
    
    return {
        'stroke_width': float(stroke_width),
        'char_height': float(char_height),
        'sharpness': float(sharpness),
        'noise_variance': float(noise_variance)
    }


def _coeff_variation(values: List[float]) -> float:
    """Compute coefficient of variation (std/mean)."""
    if len(values) < 2:
        return 0.0
    
    mean_val = np.mean(values)
    std_val = np.std(values)
    
    return std_val / (mean_val + 1e-5)
