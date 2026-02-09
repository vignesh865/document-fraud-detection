"""
Boundary Analyzer

Detects pasted content by analyzing region boundaries.
Pasted text has artificially sharp boundaries.
"""

import cv2
import numpy as np
from typing import Dict, List

def analyze_boundaries(region_map: np.ndarray, image: np.ndarray) -> Dict:
    """
    Analyze boundaries between regions for paste detection.
    
    Args:
        region_map: HxW array with region IDs
        image: Original image (BGR)
        
    Returns:
        Boundary analysis results
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Find region boundaries
    # Boundaries = pixels where adjacent pixels have different region IDs
    boundaries = _find_boundaries(region_map)
    
    if np.sum(boundaries) < 10:
        return {
            'is_suspicious': False,
            'reason': 'Insufficient boundaries to analyze'
        }
    
    # Compute gradient magnitude at boundaries
    gradient_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    
    # Extract gradient values at boundaries
    boundary_gradients = gradient_magnitude[boundaries]
    
    # Compute statistics
    mean_gradient = np.mean(boundary_gradients)
    max_gradient = np.max(boundary_gradients)
    std_gradient = np.std(boundary_gradients)
    
    # Detect sharp boundaries
    # Pasted content has very sharp, artificial edges
    sharp_threshold = mean_gradient + 2 * std_gradient
    sharp_boundaries = boundary_gradients > sharp_threshold
    sharp_ratio = np.sum(sharp_boundaries) / len(boundary_gradients)
    
    # Decision
    # High ratio of sharp boundaries suggests pasting
    # Real passport photo pages have sharp photo edges - this is NORMAL
    # Only flag if VERY high ratio or extremely sharp
    is_suspicious = (sharp_ratio > 0.35) or (max_gradient > 600)  # Was 0.2 / 200
    
    return {
        'is_suspicious': bool(is_suspicious),
        'mean_gradient': float(mean_gradient),
        'max_gradient': float(max_gradient),
        'sharp_ratio': float(sharp_ratio),
        'num_boundaries': int(np.sum(boundaries)),
        'interpretation': 'Sharp boundaries (>20%) suggest pasted content with unnatural edges'
    }


def _find_boundaries(region_map: np.ndarray) -> np.ndarray:
    """
    Find pixels at region boundaries.
    
    A pixel is a boundary if any of its 4-neighbors has a different region ID.
    """
    h, w = region_map.shape
    boundaries = np.zeros((h, w), dtype=bool)
    
    # Check 4-connectivity
    # Right neighbor
    boundaries[:-1, :] |= (region_map[:-1, :] != region_map[1:, :])
    # Bottom neighbor
    boundaries[:, :-1] |= (region_map[:, :-1] != region_map[:, 1:])
    # Left neighbor
    boundaries[1:, :] |= (region_map[1:, :] != region_map[:-1, :])
    # Top neighbor
    boundaries[:, 1:] |= (region_map[:, 1:] != region_map[:, :-1])
    
    return boundaries
