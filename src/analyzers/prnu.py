
import cv2
import numpy as np
import os
from skimage.restoration import denoise_wavelet

def extract_noise_pattern(image_path, output_path=None):
    """
    Extracts the Photo Response Non-Uniformity (PRNU) noise residual.
    This is a simplified implementation that extracts the noise residual 
    using wavelet denoising.
    
    In a full forensic scenario, this residual is compared against a reference 
    camera fingerprint. Here, we return the residual statistics, as 
    AI images often lack the specific sensor noise structure of real cameras.
    
    Args:
        image_path (str): Path to the image.
        output_path (str): Path to save the extracted noise pattern.
        
    Returns:
        dict: Noise statistics.
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            return {"error": "Could not read image"}
            
        # Convert to grayscale for simpler analysis
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
            
        # Ensure float type for processing
        gray = gray.astype(np.float32) / 255.0

        # Denoise using Wavelet to estimate the "clean" image
        denoised = denoise_wavelet(gray, channel_axis=None, rescale_sigma=True)
        
        # Calculate Residual (Noise = Original - Denoised)
        noise_residual = gray - denoised
        
        # Normalize residual for visualization [0, 255]
        noise_vis = cv2.normalize(noise_residual, None, 0, 255, cv2.NORM_MINMAX)
        noise_vis = noise_vis.astype(np.uint8)
        
        if output_path:
            # Enhance contrast for better visibility
            noise_vis_enhanced = cv2.equalizeHist(noise_vis)
            cv2.imwrite(output_path, noise_vis_enhanced)
            print(f"PRNU extraction saved to {output_path}")
            
        # Metrics
        # AI images might have very low variance in this residual (too clean)
        # or specific periodic patterns. Real cameras have random but characteristic noise.
        std_dev = np.std(noise_residual)
        variance = np.var(noise_residual)
        
        # Heuristic Threshold
        # Real sensors usually have noise variance > 0.001 (normalized 0-1) or > 1.0 (0-255) depending on ISO.
        # "Perfectly clean" synthetic images often have variance near 0.
        # WE used 0-1 scale for calculation above (gray = img/255.0).
        # Threshold: if variance is TOO low, it's suspicious (synthetic).
        is_suspicious = variance < 1e-5 

        return {
            "noise_std_dev": float(std_dev),
            "noise_variance": float(variance),
            "is_suspicious": is_suspicious,
            "interpretation": "Extremely low variance (< 1e-5) indicates synthetic generation (too clean)."
        }

    except Exception as e:
        return {"error": str(e)}
