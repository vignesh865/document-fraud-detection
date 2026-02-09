
"""
Noise Variance Analysis Module
Detects manipulation by analyzing the noise distribution in the image.
Spliced or edited regions often have inconsistent noise patterns compared to the rest of the image.
"""

import cv2
import numpy as np


def analyze_noise(image_path, output_path=None):
    """
    Performs noise analysis on an image.
    
    Args:
        image_path (str): Path to the source image.
        output_path (str, optional): Path to save the noise map visualization.
        
    Returns:
        numpy.ndarray: The noise map.
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error loading image {image_path}")
        return None

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply a denoising filter (Median Blur is simple and effective for this)
    # Alternatively, Non-Local Means Denoising could be used for better quality but slower speed
    denoised = cv2.medianBlur(gray, 3)

    # Calculate the absolute difference between the original gray image and the denoised version
    noise_map = cv2.absdiff(gray, denoised)

    # Enhance the noise map for better visibility (optional, for visualization)
    # We can normalize it to 0-255 range
    noise_map_normalized = cv2.normalize(noise_map, None, 0, 255, cv2.NORM_MINMAX)
    
    if output_path:
         # Apply a colormap for better visualization
        noise_map_color = cv2.applyColorMap(noise_map_normalized, cv2.COLORMAP_JET)
        cv2.imwrite(output_path, noise_map_color)
        print(f"Noise map saved to {output_path}")

    return noise_map_normalized

