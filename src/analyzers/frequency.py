import cv2
import numpy as np
import os
import matplotlib
matplotlib.use('Agg') # Use non-interactive backend
import matplotlib.pyplot as plt

def analyze_frequency(image_path, output_path=None):
    """
    Analyzes the frequency domain of an image using Fourier Transform.
    GAN-generated images often exhibit specific artifacts in the frequency domain
    (e.g., checkerboard patterns, heavy high-frequency components).
    
    Args:
        image_path (str): Path to the image.
        output_path (str): Path to save the frequency spectrum plot.
        
    Returns:
        dict: Metrics derived from the frequency spectrum.
    """
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return {"error": "Could not read image"}

        # 1. Compute DFT
        f = np.fft.fft2(img)
        fshift = np.fft.fftshift(f)
        
        # 2. Compute Magnitude Spectrum (log scale)
        magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
        
        # 3. Analyze for anomalies
        # Real photos usually have a smooth fall-off of energy from low to high frequencies (1/f statistic).
        # Deepfakes might have spikes or "grid" patterns.
        
        # Simple metric: Ratio of high frequency energy to low frequency energy
        h, w = magnitude_spectrum.shape
        cy, cx = h // 2, w // 2
        radius_low = min(h, w) // 8
        
        # Create masks
        y, x = np.ogrid[:h, :w]
        mask_area = (x - cx)**2 + (y - cy)**2 <= radius_low**2
        
        low_freq_energy = np.mean(magnitude_spectrum[mask_area])
        high_freq_energy = np.mean(magnitude_spectrum[~mask_area])
        
        ratio = high_freq_energy / (low_freq_energy + 1e-5)
        
        # Heuristic Threshold
        # Normal images usually have much more low-freq energy.
        # Deepfakes/GANs often have high-freq artifacts (checkerboard).
        # Threshold is experimental.
        is_suspicious = ratio > 0.5

        if output_path:
            plt.figure(figsize=(10, 5))
            plt.subplot(121), plt.imshow(img, cmap='gray')
            plt.title('Input Image'), plt.axis('off')
            plt.subplot(122), plt.imshow(magnitude_spectrum, cmap='gray')
            plt.title('Magnitude Spectrum'), plt.axis('off')
            plt.savefig(output_path)
            plt.close()
            print(f"Frequency analysis plot saved to {output_path}")

        return {
            "high_freq_energy": float(high_freq_energy),
            "low_freq_energy": float(low_freq_energy),
            "energy_ratio": float(ratio),
            "is_suspicious": is_suspicious,
            "interpretation": "High energy ratio (> 0.5) indicates abnormal high-frequency patterns typical of GANs."
        }

    except Exception as e:
        return {"error": str(e)}
