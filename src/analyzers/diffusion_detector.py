"""
Diffusion Model Detector - Proof of Concept

This implements detection techniques specifically designed for diffusion models
like Stable Diffusion, DALL-E, Midjourney, etc.

WARNING: These techniques are DIFFERENT from GAN detection.
Do NOT use the GAN detector for diffusion model outputs!
"""

import cv2
import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats, ndimage
from scipy.signal import correlate2d
from typing import Dict, Optional
import warnings
warnings.filterwarnings('ignore')


class DiffusionModelDetector:
    """
    Specialized detector for diffusion model outputs.
    
    Detects artifacts specific to iterative denoising-based generation:
    1. Local frequency inconsistencies (patch-based analysis)
    2. Texture repetition patterns
    3. Noise residual characteristics
    4. Cross-channel frequency correlation
    5. Multi-scale spectral inconsistencies
    """
    
    def __init__(self):
        self.results = {}
        
    def analyze_image(self, image_path: str, output_dir: Optional[str] = None) -> Dict:
        """
        Analyze image for diffusion model artifacts.
        
        Args:
            image_path: Path to image file
            output_dir: Directory to save visualizations (optional)
            
        Returns:
            Dictionary containing analysis results and verdict
        """
        try:
            # Read image
            img = cv2.imread(image_path)
            if img is None:
                return {"error": "Could not read image"}
            
            # Convert to grayscale for some analyses
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Perform all analyses
            self.results = {
                "image_shape": gray.shape,
                "local_frequency_inconsistency": 
                    self._detect_local_frequency_inconsistencies(gray),
                "texture_repetition": 
                    self._detect_texture_repetition(gray),
                "noise_residual_analysis": 
                    self._analyze_noise_residuals(gray),
                "cross_channel_correlation": 
                    self._analyze_cross_channel_correlation(img),
                "multi_scale_spectral": 
                    self._multi_scale_spectral_analysis(gray),
            }
            
            # Compute final verdict
            self.results["verdict"] = self._compute_verdict()
            
            # Generate visualizations if requested
            if output_dir:
                self._generate_visualizations(gray, img, image_path, output_dir)
            
            return self.results
            
        except Exception as e:
            return {"error": str(e)}
    
    def _detect_local_frequency_inconsistencies(self, image: np.ndarray) -> Dict:
        """
        Detect inconsistencies in local frequency distributions.
        
        Diffusion models synthesize different regions somewhat independently,
        leading to inconsistent local frequency characteristics that wouldn't
        appear in natural images.
        
        Returns:
            Dictionary with local frequency analysis results
        """
        h, w = image.shape
        patch_size = 64
        overlap = patch_size // 2
        
        # Extract patches and compute frequency features
        patch_alphas = []
        patch_energies = []
        patch_positions = []
        
        for i in range(0, h - patch_size, overlap):
            for j in range(0, w - patch_size, overlap):
                patch = image[i:i+patch_size, j:j+patch_size]
                
                # Compute FFT
                f = np.fft.fft2(patch)
                fshift = np.fft.fftshift(f)
                magnitude = np.abs(fshift)
                
                # Compute power law exponent for this patch
                cy, cx = patch_size // 2, patch_size // 2
                y, x = np.ogrid[:patch_size, :patch_size]
                r = np.sqrt((x - cx)**2 + (y - cy)**2)
                
                # Radial average
                max_radius = patch_size // 2
                radii = np.arange(1, max_radius)
                radial_psd = []
                
                for radius in radii:
                    mask = (r >= radius) & (r < radius + 1)
                    if np.any(mask):
                        radial_psd.append(np.mean(magnitude[mask] ** 2))
                
                radial_psd = np.array(radial_psd)
                
                # Fit power law
                if len(radial_psd) > 10 and np.all(radial_psd > 0):
                    log_r = np.log(radii[:len(radial_psd)])
                    log_psd = np.log(radial_psd)
                    slope, _, _, _, _ = stats.linregress(log_r, log_psd)
                    alpha = -slope
                    patch_alphas.append(alpha)
                    patch_energies.append(np.mean(magnitude))
                    patch_positions.append((i, j))
        
        if len(patch_alphas) < 10:
            return {
                "inconsistency_score": 0.0,
                "is_suspicious": False,
                "interpretation": "Insufficient patches for analysis"
            }
        
        patch_alphas = np.array(patch_alphas)
        patch_energies = np.array(patch_energies)
        
        # Compute variance in alpha across patches
        # Natural images: consistent α across regions
        # Diffusion models: high variance (different synthesis quality)
        alpha_variance = np.var(patch_alphas)
        energy_variance = np.var(patch_energies) / (np.mean(patch_energies) + 1e-5)
        
        # Combined inconsistency score
        inconsistency_score = min(1.0, (alpha_variance * 2.0) + (energy_variance * 0.5))
        
        is_suspicious = inconsistency_score > 0.4
        
        return {
            "inconsistency_score": float(inconsistency_score),
            "alpha_variance": float(alpha_variance),
            "energy_variance": float(energy_variance),
            "num_patches": len(patch_alphas),
            "mean_alpha": float(np.mean(patch_alphas)),
            "is_suspicious": bool(is_suspicious),
            "interpretation": "High score (>0.4) suggests inconsistent local synthesis"
        }
    
    def _detect_texture_repetition(self, image: np.ndarray) -> Dict:
        """
        Detect repeated texture patterns via autocorrelation.
        
        Diffusion models sometimes repeat similar texture patterns across
        different regions, creating detectable autocorrelation peaks.
        
        Returns:
            Dictionary with texture repetition analysis
        """
        # Compute FFT
        f = np.fft.fft2(image)
        fshift = np.fft.fftshift(f)
        magnitude = np.abs(fshift)
        
        # Compute autocorrelation of magnitude spectrum
        # High autocorrelation peaks indicate repeated patterns
        autocorr = correlate2d(magnitude, magnitude, mode='same')
        autocorr = autocorr / np.max(autocorr)  # Normalize
        
        h, w = autocorr.shape
        cy, cx = h // 2, w // 2
        
        # Mask out the center peak
        y, x = np.ogrid[:h, :w]
        center_mask = np.sqrt((x - cx)**2 + (y - cy)**2) < min(h, w) * 0.1
        autocorr_masked = autocorr.copy()
        autocorr_masked[center_mask] = 0
        
        # Find secondary peaks
        # Use a threshold for peak detection
        peak_threshold = 0.3
        peaks = autocorr_masked > peak_threshold
        num_peaks = np.sum(peaks)
        
        # Compute repetition score
        max_secondary_peak = np.max(autocorr_masked) if num_peaks > 0 else 0.0
        repetition_score = min(1.0, max_secondary_peak * 2.0 + num_peaks / 100.0)
        
        is_suspicious = repetition_score > 0.35
        
        return {
            "repetition_score": float(repetition_score),
            "num_peaks": int(num_peaks),
            "max_secondary_peak": float(max_secondary_peak),
            "is_suspicious": bool(is_suspicious),
            "interpretation": "High score (>0.35) suggests texture repetition from diffusion sampling"
        }
    
    def _analyze_noise_residuals(self, image: np.ndarray) -> Dict:
        """
        Analyze high-frequency noise residuals.
        
        Diffusion models leave characteristic noise patterns from the
        iterative denoising process that differ from natural camera noise.
        
        Returns:
            Dictionary with noise residual analysis
        """
        # Extract noise residual by denoising
        denoised = ndimage.gaussian_filter(image, sigma=1.5)
        residual = image.astype(float) - denoised.astype(float)
        
        # Analyze residual in frequency domain
        f = np.fft.fft2(residual)
        fshift = np.fft.fftshift(f)
        magnitude = np.abs(fshift)
        
        # Natural camera noise: relatively uniform (white noise)
        # Diffusion residuals: structured patterns
        
        # Compute entropy of residual spectrum
        # Flatten and normalize
        mag_flat = magnitude.flatten()
        mag_flat = mag_flat / (np.sum(mag_flat) + 1e-10)
        
        # Compute histogram
        hist, _ = np.histogram(mag_flat, bins=100, density=True)
        hist = hist[hist > 0]
        
        # Entropy
        entropy = -np.sum(hist * np.log2(hist + 1e-10))
        max_entropy = np.log2(100)
        normalized_entropy = entropy / max_entropy
        
        # Also check for structure in residual
        # Compute autocorrelation of residual
        residual_autocorr = correlate2d(residual, residual, mode='same')
        residual_autocorr = residual_autocorr / np.max(np.abs(residual_autocorr))
        
        # Check for structure (high autocorr at non-zero lags)
        h, w = residual_autocorr.shape
        cy, cx = h // 2, w // 2
        y, x = np.ogrid[:h, :w]
        
        # Sample a ring (not center, not too far)
        ring_mask = (np.sqrt((x - cx)**2 + (y - cy)**2) > 5) & \
                    (np.sqrt((x - cx)**2 + (y - cy)**2) < 20)
        
        ring_autocorr = np.mean(np.abs(residual_autocorr[ring_mask]))
        
        # Score: low entropy OR high structure = suspicious
        structure_score = min(1.0, ring_autocorr * 3.0)
        entropy_score = 1.0 - normalized_entropy
        
        residual_score = (structure_score + entropy_score) / 2.0
        
        is_suspicious = residual_score > 0.5
        
        return {
            "residual_score": float(residual_score),
            "normalized_entropy": float(normalized_entropy),
            "structure_score": float(structure_score),
            "is_suspicious": bool(is_suspicious),
            "interpretation": "High score (>0.5) indicates non-natural noise residuals from denoising"
        }
    
    def _analyze_cross_channel_correlation(self, image_rgb: np.ndarray) -> Dict:
        """
        Analyze frequency correlation between RGB channels.
        
        Natural images have high correlation between color channels in
        frequency domain. Diffusion models sometimes process channels
        semi-independently, reducing correlation.
        
        Returns:
            Dictionary with cross-channel correlation analysis
        """
        if len(image_rgb.shape) != 3 or image_rgb.shape[2] != 3:
            return {
                "correlation_score": 0.0,
                "is_suspicious": False,
                "interpretation": "Not an RGB image"
            }
        
        # Compute FFT for each channel
        r_fft = np.fft.fft2(image_rgb[:,:,0])
        g_fft = np.fft.fft2(image_rgb[:,:,1])
        b_fft = np.fft.fft2(image_rgb[:,:,2])
        
        # Get magnitude
        r_mag = np.abs(np.fft.fftshift(r_fft))
        g_mag = np.abs(np.fft.fftshift(g_fft))
        b_mag = np.abs(np.fft.fftshift(b_fft))
        
        # Focus on mid-frequencies (most informative)
        h, w = r_mag.shape
        cy, cx = h // 2, w // 2
        y, x = np.ogrid[:h, :w]
        r_dist = np.sqrt((x - cx)**2 + (y - cy)**2)
        
        mid_freq_mask = (r_dist > min(h,w) * 0.1) & (r_dist < min(h,w) * 0.4)
        
        # Compute correlations in mid-frequency band
        r_flat = r_mag[mid_freq_mask].flatten()
        g_flat = g_mag[mid_freq_mask].flatten()
        b_flat = b_mag[mid_freq_mask].flatten()
        
        rg_corr = np.corrcoef(r_flat, g_flat)[0,1]
        rb_corr = np.corrcoef(r_flat, b_flat)[0,1]
        gb_corr = np.corrcoef(g_flat, b_flat)[0,1]
        
        avg_corr = (rg_corr + rb_corr + gb_corr) / 3.0
        
        # Natural images: high correlation (>0.8)
        # Diffusion models: lower correlation
        correlation_score = 1.0 - avg_corr  # Invert: low corr = high score
        
        is_suspicious = avg_corr < 0.75
        
        return {
            "correlation_score": float(correlation_score),
            "avg_correlation": float(avg_corr),
            "rg_correlation": float(rg_corr),
            "rb_correlation": float(rb_corr),
            "gb_correlation": float(gb_corr),
            "is_suspicious": bool(is_suspicious),
            "interpretation": "Low correlation (<0.75) suggests independent channel synthesis"
        }
    
    def _multi_scale_spectral_analysis(self, image: np.ndarray) -> Dict:
        """
        Analyze spectral characteristics at multiple scales.
        
        Diffusion models use different denoising strengths at different
        scales, creating scale-dependent artifacts.
        
        Returns:
            Dictionary with multi-scale analysis
        """
        scales = [1.0, 0.5, 0.25]
        scale_features = []
        
        for scale in scales:
            if scale < 1.0:
                new_h = int(image.shape[0] * scale)
                new_w = int(image.shape[1] * scale)
                scaled = cv2.resize(image, (new_w, new_h))
            else:
                scaled = image
            
            # Compute power law alpha
            f = np.fft.fft2(scaled)
            fshift = np.fft.fftshift(f)
            magnitude = np.abs(fshift)
            
            h, w = magnitude.shape
            cy, cx = h // 2, w // 2
            y, x = np.ogrid[:h, :w]
            r = np.sqrt((x - cx)**2 + (y - cy)**2)
            
            # Radial average
            max_radius = min(h, w) // 2
            radii = np.arange(1, max_radius)
            radial_psd = []
            
            for radius in radii:
                mask = (r >= radius) & (r < radius + 1)
                if np.any(mask):
                    radial_psd.append(np.mean(magnitude[mask] ** 2))
            
            radial_psd = np.array(radial_psd)
            
            # Fit power law
            if len(radial_psd) > 10 and np.all(radial_psd > 0):
                log_r = np.log(radii[:len(radial_psd)])
                log_psd = np.log(radial_psd)
                slope, _, r_val, _, _ = stats.linregress(log_r, log_psd)
                alpha = -slope
                scale_features.append({
                    'scale': scale,
                    'alpha': alpha,
                    'r_squared': r_val**2
                })
        
        if len(scale_features) < 2:
            return {
                "multi_scale_score": 0.0,
                "is_suspicious": False,
                "interpretation": "Insufficient scales for analysis"
            }
        
        # Natural images: consistent alpha across scales
        # Diffusion: inconsistent (scale-dependent synthesis)
        alphas = [f['alpha'] for f in scale_features]
        alpha_variance = np.var(alphas)
        
        multi_scale_score = min(1.0, alpha_variance * 3.0)
        
        is_suspicious = alpha_variance > 0.3
        
        return {
            "multi_scale_score": float(multi_scale_score),
            "alpha_variance": float(alpha_variance),
            "scale_features": scale_features,
            "is_suspicious": bool(is_suspicious),
            "interpretation": "High variance (>0.3) suggests scale-dependent synthesis"
        }
    
    def _compute_verdict(self) -> Dict:
        """
        Combine all analyses to produce final verdict.
        
        Returns:
            Dictionary with overall assessment
        """
        checks = {
            "local_frequency_inconsistency": 
                self.results["local_frequency_inconsistency"]["is_suspicious"],
            "texture_repetition": 
                self.results["texture_repetition"]["is_suspicious"],
            "noise_residual": 
                self.results["noise_residual_analysis"]["is_suspicious"],
            "cross_channel_correlation": 
                self.results["cross_channel_correlation"]["is_suspicious"],
            "multi_scale_spectral": 
                self.results["multi_scale_spectral"]["is_suspicious"]
        }
        
        suspicious_count = sum(checks.values())
        total_checks = len(checks)
        
        confidence = suspicious_count / total_checks
        
        # Determine verdict
        if confidence >= 0.6:
            verdict = "LIKELY_DIFFUSION_MODEL"
            explanation = f"Multiple diffusion-specific signals detected ({suspicious_count}/{total_checks})"
        elif confidence >= 0.4:
            verdict = "SUSPICIOUS"
            explanation = f"Some diffusion artifacts detected ({suspicious_count}/{total_checks}). Further analysis recommended"
        else:
            verdict = "LIKELY_REAL"
            explanation = f"Few diffusion artifacts detected ({suspicious_count}/{total_checks}). Appears natural"
        
        return {
            "verdict": verdict,
            "confidence": float(confidence),
            "suspicious_signals": suspicious_count,
            "total_checks": total_checks,
            "detailed_checks": checks,
            "explanation": explanation,
            "disclaimer": "Specialized for diffusion models (Stable Diffusion, DALL-E, Midjourney). For GANs, use the GAN detector. Modern diffusion models are VERY hard to detect - this is probabilistic, not definitive."
        }
    
    def _generate_visualizations(self, gray_image, rgb_image, image_path, output_dir):
        """Generate visualization plots."""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        fig = plt.figure(figsize=(18, 10))
        
        # 1. Original image
        plt.subplot(2, 4, 1)
        plt.imshow(cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB))
        plt.title('Original Image', fontsize=12, fontweight='bold')
        plt.axis('off')
        
        # 2. Local frequency inconsistency
        plt.subplot(2, 4, 2)
        lf = self.results["local_frequency_inconsistency"]
        text = f'Local Frequency\nInconsistency\n\nScore: {lf["inconsistency_score"]:.3f}\n'
        text += f'Alpha Var: {lf["alpha_variance"]:.3f}\n'
        text += f'Status: {"⚠️  SUSPICIOUS" if lf["is_suspicious"] else "✓ PASS"}'
        plt.text(0.5, 0.5, text, ha='center', va='center', fontsize=11,
                bbox=dict(boxstyle='round', facecolor='lightyellow' if lf["is_suspicious"] else 'lightgreen'))
        plt.title('Test 1', fontsize=12, fontweight='bold')
        plt.axis('off')
        
        # 3. Texture repetition
        plt.subplot(2, 4, 3)
        tr = self.results["texture_repetition"]
        text = f'Texture Repetition\n\nScore: {tr["repetition_score"]:.3f}\n'
        text += f'Peaks: {tr["num_peaks"]}\n'
        text += f'Status: {"⚠️  SUSPICIOUS" if tr["is_suspicious"] else "✓ PASS"}'
        plt.text(0.5, 0.5, text, ha='center', va='center', fontsize=11,
                bbox=dict(boxstyle='round', facecolor='lightyellow' if tr["is_suspicious"] else 'lightgreen'))
        plt.title('Test 2', fontsize=12, fontweight='bold')
        plt.axis('off')
        
        # 4. Noise residual
        plt.subplot(2, 4, 4)
        nr = self.results["noise_residual_analysis"]
        text = f'Noise Residual\n\nScore: {nr["residual_score"]:.3f}\n'
        text += f'Entropy: {nr["normalized_entropy"]:.3f}\n'
        text += f'Status: {"⚠️  SUSPICIOUS" if nr["is_suspicious"] else "✓ PASS"}'
        plt.text(0.5, 0.5, text, ha='center', va='center', fontsize=11,
                bbox=dict(boxstyle='round', facecolor='lightyellow' if nr["is_suspicious"] else 'lightgreen'))
        plt.title('Test 3', fontsize=12, fontweight='bold')
        plt.axis('off')
        
        # 5. Cross-channel correlation
        plt.subplot(2, 4, 5)
        cc = self.results["cross_channel_correlation"]
        text = f'Cross-Channel\nCorrelation\n\nAvg Corr: {cc["avg_correlation"]:.3f}\n'
        text += f'Status: {"⚠️  SUSPICIOUS" if cc["is_suspicious"] else "✓ PASS"}'
        plt.text(0.5, 0.5, text, ha='center', va='center', fontsize=11,
                bbox=dict(boxstyle='round', facecolor='lightyellow' if cc["is_suspicious"] else 'lightgreen'))
        plt.title('Test 4', fontsize=12, fontweight='bold')
        plt.axis('off')
        
        # 6. Multi-scale
        plt.subplot(2, 4, 6)
        ms = self.results["multi_scale_spectral"]
        text = f'Multi-Scale\nSpectral\n\nAlpha Var: {ms["alpha_variance"]:.3f}\n'
        text += f'Status: {"⚠️  SUSPICIOUS" if ms["is_suspicious"] else "✓ PASS"}'
        plt.text(0.5, 0.5, text, ha='center', va='center', fontsize=11,
                bbox=dict(boxstyle='round', facecolor='lightyellow' if ms["is_suspicious"] else 'lightgreen'))
        plt.title('Test 5', fontsize=12, fontweight='bold')
        plt.axis('off')
        
        # 7. Overall verdict
        plt.subplot(2, 4, 7)
        v = self.results["verdict"]
        verdict_text = f'VERDICT\n\n{v["verdict"]}\n\n'
        verdict_text += f'Confidence: {v["confidence"]:.1%}\n'
        verdict_text += f'Signals: {v["suspicious_signals"]}/{v["total_checks"]}'
        
        if v["verdict"] == "LIKELY_DIFFUSION_MODEL":
            color = 'lightcoral'
        elif v["verdict"] == "SUSPICIOUS":
            color = 'lightyellow'
        else:
            color = 'lightgreen'
        
        plt.text(0.5, 0.5, verdict_text, ha='center', va='center', 
                fontsize=14, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor=color, alpha=0.8))
        plt.title('Final Verdict', fontsize=12, fontweight='bold')
        plt.axis('off')
        
        # 8. Explanation
        plt.subplot(2, 4, 8)
        expl = v["explanation"] + "\n\n" + v["disclaimer"]
        plt.text(0.05, 0.95, expl, ha='left', va='top', fontsize=8, wrap=True,
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
        plt.title('Explanation', fontsize=12, fontweight='bold')
        plt.axis('off')
        
        plt.tight_layout()
        
        output_path = os.path.join(output_dir, 'diffusion_analysis.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"\n✓ Visualization saved to: {output_path}")


def main():
    """Example usage."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python diffusion_detector.py <image_path> [output_dir]")
        sys.exit(1)
    
    image_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "./diffusion_analysis_output"
    
    print("="*80)
    print("DIFFUSION MODEL DETECTOR")
    print("="*80)
    print(f"\nAnalyzing: {image_path}\n")
    
    detector = DiffusionModelDetector()
    results = detector.analyze_image(image_path, output_dir)
    
    if "error" in results:
        print(f"❌ Error: {results['error']}")
        sys.exit(1)
    
    # Print results
    print("\n" + "="*80)
    print("DIFFUSION MODEL ANALYSIS RESULTS")
    print("="*80)
    
    for i, (test_name, test_result) in enumerate([
        ("LOCAL FREQUENCY INCONSISTENCY", results["local_frequency_inconsistency"]),
        ("TEXTURE REPETITION", results["texture_repetition"]),
        ("NOISE RESIDUAL ANALYSIS", results["noise_residual_analysis"]),
        ("CROSS-CHANNEL CORRELATION", results["cross_channel_correlation"]),
        ("MULTI-SCALE SPECTRAL", results["multi_scale_spectral"])
    ], 1):
        print(f"\n{i}. {test_name}")
        print("-" * 40)
        print(f"   Status: {'⚠️  SUSPICIOUS' if test_result['is_suspicious'] else '✓ PASS'}")
        for key, val in test_result.items():
            if key not in ['is_suspicious', 'interpretation']:
                print(f"   {key}: {val}")
        print(f"   {test_result['interpretation']}")
    
    print("\n" + "="*80)
    print("FINAL VERDICT")
    print("="*80)
    v = results["verdict"]
    print(f"\n   Verdict: {v['verdict']}")
    print(f"   Confidence: {v['confidence']:.1%}")
    print(f"   Suspicious signals: {v['suspicious_signals']}/{v['total_checks']}")
    print(f"\n   {v['explanation']}")
    print(f"\n   ⚠️  {v['disclaimer']}")
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
