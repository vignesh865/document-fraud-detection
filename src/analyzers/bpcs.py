
import cv2
import numpy as np
import os

def analyze_bpcs(image_path, output_path=None):
    """
    Bit-Plane Complexity Segmentation (BPCS)
    Analyzes the complexity of bit planes.
    
    Args:
        image_path (str): Path to image.
        output_path (str): Path to save visualization.
        
    Returns:
        dict: Complexity metrics.
    """
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return {"error": "Could not read image"}
            
        # Extract bit planes
        planes = []
        for i in range(8):
            # Extract i-th bit
            plane = (img >> i) & 1
            planes.append(plane * 255)
            
        # For simplicity, we save the first 4 (LSB) planes combined or separately
        # LSB is plane 0.
        
        lsb_plane = planes[0].astype(np.uint8)
        
        if output_path:
            cv2.imwrite(output_path, lsb_plane)
            print(f"BPCS LSB plane saved to {output_path}")
            
        # Calculate complexity (simple transition count)
        # A more complex metric would be 'complexity' as defined in BPCS steganography papers
        # Here we just return the sum of transitions as a rough proxy for 'noise'
        
        transitions_x = np.sum(np.abs(np.diff(lsb_plane, axis=1)))
        transitions_y = np.sum(np.abs(np.diff(lsb_plane, axis=0)))
        
        complexity = (transitions_x + transitions_y) / (lsb_plane.size)
        
        return {
            "lsb_complexity": complexity,
            "interpretation": "High complexity in LSB usually indicates random noise (normal for photos). Low complexity or blocks of low complexity might indicate tampering or computer generated regions."
        }

    except Exception as e:
        return {"error": str(e)}
