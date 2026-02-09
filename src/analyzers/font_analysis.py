"""
Font Consistency Analysis

Detects when different fonts/text rendering methods are used in a document.
Common in document fraud when someone pastes text onto a scanned form.
"""

import cv2
import numpy as np
from scipy import ndimage
from typing import Dict, List, Tuple

def analyze_font_consistency(image_path: str, output_path: str = None) -> Dict:
    """
    Analyze font consistency across text regions in a document.
    
    Strategy:
    1. Detect text regions using edge detection + morphological operations
    2. For each text region, measure font characteristics:
       - Stroke width
       - Character height distribution
       - Spacing (kerning)
    3. Flag regions with outlier characteristics
    
    Args:
        image_path: Path to document image
        output_path: Optional path to save visualization
        
    Returns:
        Dictionary with analysis results
    """
    try:
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            return {"error": "Could not read image"}
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Detect text regions
        text_regions = _detect_text_regions(gray)
        
        if len(text_regions) < 2:
            return {
                "is_suspicious": False,
                "num_regions": len(text_regions),
                "interpretation": "Insufficient text regions for comparison"
            }
        
        # Analyze each region
        region_features = []
        for i, (x, y, w, h) in enumerate(text_regions):
            region = gray[y:y+h, x:x+w]
            features = _extract_font_features(region)
            features['bbox'] = (x, y, w, h)
            region_features.append(features)
        
        # Compute consistency metrics
        stroke_widths = [f['stroke_width'] for f in region_features]
        char_heights = [f['char_height'] for f in region_features]
        
        # Coefficient of variation (std/mean)
        stroke_width_cv = np.std(stroke_widths) / (np.mean(stroke_widths) + 1e-5)
        char_height_cv = np.std(char_heights) / (np.mean(char_heights) + 1e-5)
        
        # Consistency score (low CV = consistent, high CV = inconsistent)
        consistency_score = (stroke_width_cv + char_height_cv) / 2.0
        
        # Threshold: CV > 0.3 is suspicious for documents (should be uniform)
        is_suspicious = consistency_score > 0.3
        
        # Find outlier regions
        outliers = []
        stroke_mean = np.mean(stroke_widths)
        stroke_std = np.std(stroke_widths)
        
        for i, features in enumerate(region_features):
            z_score = abs(features['stroke_width'] - stroke_mean) / (stroke_std + 1e-5)
            if z_score > 2.0:  # More than 2 std deviations
                outliers.append({
                    'region_id': i,
                    'bbox': features['bbox'],
                    'z_score': float(z_score),
                    'stroke_width': features['stroke_width']
                })
        
        results = {
            "is_suspicious": bool(is_suspicious),
            "consistency_score": float(consistency_score),
            "num_regions": len(text_regions),
            "num_outliers": len(outliers),
            "outlier_regions": outliers,
            "stroke_width_cv": float(stroke_width_cv),
            "char_height_cv": float(char_height_cv),
            "interpretation": f"High CV (>{0.3:.1f}) suggests multiple fonts/rendering methods"
        }
        
        # Visualization
        if output_path:
            _visualize_font_analysis(img, region_features, outliers, output_path)
        
        return results
        
    except Exception as e:
        return {"error": str(e)}


def _detect_text_regions(gray: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """
    Detect text regions using edge detection + morphological operations.
    
    Returns:
        List of bounding boxes (x, y, w, h)
    """
    # Apply adaptive threshold to handle varying lighting
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 15, 5
    )
    
    # Morphological operations to connect text
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 3))
    dilated = cv2.dilate(binary, kernel, iterations=2)
    
    # Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours by size
    text_regions = []
    h, w = gray.shape
    min_area = (h * w) * 0.001  # At least 0.1% of image
    
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        
        # Filter: reasonable size, not too thin/wide
        aspect_ratio = w / float(h) if h > 0 else 0
        if area > min_area and 0.5 < aspect_ratio < 20:
            text_regions.append((x, y, w, h))
    
    return text_regions


def _extract_font_features(region: np.ndarray) -> Dict:
    """
    Extract font characteristics from a text region.
    
    Returns:
        Dictionary with font features
    """
    # Binarize
    _, binary = cv2.threshold(region, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Stroke width estimation via distance transform
    dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    stroke_width = np.mean(dist_transform[dist_transform > 0]) * 2 if np.any(dist_transform > 0) else 0
    
    # Character height estimation
    # Find connected components (individual characters)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    
    if num_labels > 1:
        # Exclude background (label 0)
        heights = stats[1:, cv2.CC_STAT_HEIGHT]
        char_height = np.median(heights) if len(heights) > 0 else 0
    else:
        char_height = 0
    
    return {
        'stroke_width': float(stroke_width),
        'char_height': float(char_height),
    }


def _visualize_font_analysis(img: np.ndarray, region_features: List[Dict], 
                              outliers: List[Dict], output_path: str):
    """Generate visualization showing text regions and outliers."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    vis_img = img.copy()
    
    # Draw all regions in green
    for features in region_features:
        x, y, w, h = features['bbox']
        cv2.rectangle(vis_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    # Highlight outliers in red
    for outlier in outliers:
        x, y, w, h = outlier['bbox']
        cv2.rectangle(vis_img, (x, y), (x+w, y+h), (0, 0, 255), 3)
        cv2.putText(vis_img, f"Z={outlier['z_score']:.1f}", 
                    (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    # Save
    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB))
    plt.title('Font Consistency Analysis\n(Green=Normal, Red=Outlier)')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Font analysis visualization saved to {output_path}")
