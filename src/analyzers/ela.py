
"""
Error Level Analysis (ELA) Module
Detects manipulation by re-saving an image at a specific quality and comparing it to the original.
Areas that have been edited will likely have different compression artifacts than the rest of the image.
"""

import cv2
import numpy as np
from PIL import Image, ImageChops, ImageEnhance
import os

def perform_ela(image_path, quality=90, output_path=None):
    """
    Generates an ELA image.

    Args:
        image_path (str): Path to the source image.
        quality (int): Quality level for re-saving (default: 90).
        output_path (str, optional): Path to save the ELA image.

    Returns:
        PIL.Image: The resulting ELA image.
    """
    try:
        original = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"Error opening image {image_path}: {e}")
        return None

    # Save the image at the specified quality to a temporary buffer/file
    temp_filename = 'temp_ela.jpg'
    original.save(temp_filename, 'JPEG', quality=quality)
    
    # Open the re-saved image
    resaved = Image.open(temp_filename).convert('RGB')

    # Calculate the difference between the original and re-saved image
    ela_image = ImageChops.difference(original, resaved)

    # Calculate the extrema (max difference) to scale the brightness
    extrema = ela_image.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    
    # If the image is exactly the same (no difference), max_diff will be 0
    if max_diff == 0:
        max_diff = 1 # Avoid division by zero

    scale = 255.0 / max_diff
    ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)

    # Cleanup
    os.remove(temp_filename)
    
    if output_path:
        ela_image.save(output_path)
        print(f"ELA image saved to {output_path}")

    return ela_image
