"""
Optimized Diffusion Model Detector with Progress Bars

PERFORMANCE OPTIMIZATIONS:
- Automatic image downsampling for large images
- Efficient patch sampling (not all patches)
- Optimized FFT operations
- Progress bars with tqdm
- Multi-threaded where possible
"""

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import signal, stats, ndimage
from typing import Dict, Tuple, Optional
import warnings
import time
from tqdm import tqdm
warnings.filterwarnings('ignore')


class OptimizedDiffusionDetector:
    """
    Fast diffusion model detector with automatic image resizing
    and progress tracking.
    """
    
    def __init__(self, max_dimension=1024, verbose=True):
        """
        Args:
            max_dimension: Maximum image dimension (larger images are downsampled)
            verbose: Show progress bars and timing info
        """
        self.max_dimension = max_dimension
        self.verbose = verbose
        self.results = {}
        
    def analyze_image(self, image_path: str, output_dir: Optional[str] = None) -> Dict:
        """Analyze image for diffusion model artifacts with progress tracking."""
        try:
            start_time = time.time()
            
            # Read image
            if self.verbose:
                print("📁 Loading image...")
            img = cv2.imread(image_path)
            if img is None:
                return {"error": "Could not read image"}
            
            original_shape = img.shape
            
            # Resize if needed
            img, resize_factor = self._resize_if_needed(img)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            if self.verbose:
                print(f"✓ Image loaded: {original_shape[:2]} → {img.shape[:2]} (factor: {resize_factor:.2f})")
                print(f"\n{'='*60}")
                print("Running Analysis (5 tests)")
                print(f"{'='*60}\n")
            
            # Perform analyses with progress bar
            tests = [
                ("Local Frequency Inconsistency", self._detect_local_frequency_inconsistencies),
                ("Texture Repetition", self._detect_texture_repetition),
                ("Noise Residual Analysis", self._analyze_noise_residuals),
                ("Cross-Channel Correlation", self._analyze_cross_channel_correlation),
                ("Multi-Scale Spectral", self._multi_scale_spectral_analysis)
            ]
            
            self.results = {
                "original_shape": original_shape,
                "processed_shape": img.shape,
                "resize_factor": resize_factor
            }
            
            if self.verbose:
                pbar = tqdm(tests, desc="Progress", ncols=80)
            else:
                pbar = tests
            
            for test_name, test_func in pbar:
                if self.verbose and hasattr(pbar, 'set_description'):
                    pbar.set_description(f"Running: {test_name}")
                
                test_start = time.time()
                
                # Run appropriate test
                if "cross_channel" in test_name.lower():
                    result = test_func(img)
                else:
                    result = test_func(gray)
                
                test_time = time.time() - test_start
                result['processing_time'] = test_time
                
                # Store result
                key = test_name.lower().replace(' ', '_').replace('-', '_')
                self.results[key] = result
            
            if self.verbose and hasattr(pbar, 'close'):
                pbar.close()
            
            # Compute verdict
            self.results["verdict"] = self._compute_verdict()
            
            total_time = time.time() - start_time
            self.results["total_processing_time"] = total_time
            
            if self.verbose:
                print(f"\n✓ Analysis complete in {total_time:.2f}s")
            
            # Generate visualizations
            if output_dir:
                if self.verbose:
                    print(f"\n📊 Generating visualizations...")
                self._generate_visualizations(gray, img, image_path, output_dir)
            
            return self.results
            
        except Exception as e:
            return {"error": str(e)}
    
    def _resize_if_needed(self, img: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Resize image if larger than max_dimension.
        Returns: (resized_image, resize_factor)
        """
        h, w = img.shape[:2]
        max_dim = max(h, w)
        
        if max_dim <= self.max_dimension:
            return img, 1.0
        
        # Calculate resize factor
        factor = self.max_dimension / max_dim
        new_h = int(h * factor)
        new_w = int(w * factor)
        
        # Resize with high-quality interpolation
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        return resized, factor
    
    def _detect_local_frequency_inconsistencies(self, image: np.ndarray) -> Dict:
        """Optimized local frequency analysis with sampling."""
        h, w = image.shape
        patch_size = 64
        
        # OPTIMIZATION: Sample patches instead of processing all
        max_patches = 500  # Limit number of patches
        step = max(patch_size, int(np.sqrt(h * w / max_patches)))
        
        patch_alphas = []
        patch_energies = []
        
        for i in range(0, h - patch_size, step):
            for j in range(0, w - patch_size, step):
                if len(patch_alphas) >= max_patches:
                    break
                
                patch = image[i:i+patch_size, j:j+patch_size]
                
                # Compute FFT (optimized)
                f = np.fft.rfft2(patch)  # Use rfft2 for real input (2x faster)
                magnitude = np.abs(f)
                
                # Quick power law estimate (simplified)
                cy = patch_size // 2
                cx = f.shape[1] // 2
                y, x = np.ogrid[:f.shape[0], :f.shape[1]]
                r = np.sqrt((x - cx)**2 + (y - cy)**2)
                
                # Simplified radial average (faster)
                radii = np.arange(1, min(20, f.shape[1]))  # Limit range
                radial_psd = []
                
                for radius in radii:
                    mask = (r >= radius) & (r < radius + 1)
                    if np.any(mask):
                        radial_psd.append(np.mean(magnitude[mask] ** 2))
                
                if len(radial_psd) > 5:
                    radial_psd = np.array(radial_psd)
                    log_r = np.log(radii[:len(radial_psd)])
                    log_psd = np.log(radial_psd + 1e-10)
                    
                    # Quick linear fit
                    slope = np.polyfit(log_r, log_psd, 1)[0]
                    alpha = -slope
                    
                    patch_alphas.append(alpha)
                    patch_energies.append(np.mean(magnitude))
            
            if len(patch_alphas) >= max_patches:
                break
        
        if len(patch_alphas) < 5:
            return {
                "inconsistency_score": 0.0,
                "is_suspicious": False,
                "interpretation": "Insufficient patches"
            }
        
        patch_alphas = np.array(patch_alphas)
        patch_energies = np.array(patch_energies)
        
        alpha_variance = np.var(patch_alphas)
        energy_variance = np.var(patch_energies) / (np.mean(patch_energies) + 1e-5)
        
        inconsistency_score = min(1.0, (alpha_variance * 2.0) + (energy_variance * 0.5))
        is_suspicious = inconsistency_score > 0.4
        
        return {
            "inconsistency_score": float(inconsistency_score),
            "alpha_variance": float(alpha_variance),
            "energy_variance": float(energy_variance),
            "num_patches": len(patch_alphas),
            "is_suspicious": bool(is_suspicious),
            "interpretation": "High score (>0.4) suggests inconsistent local synthesis"
        }
    
    def _detect_texture_repetition(self, image: np.ndarray) -> Dict:
        """Optimized texture repetition with downsampling."""
        # OPTIMIZATION: Downsample for autocorrelation (very expensive operation)
        max_size = 512
        h, w = image.shape
        
        if max(h, w) > max_size:
            scale = max_size / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            image_small = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            image_small = image
        
        # Compute FFT
        f = np.fft.rfft2(image_small)
        magnitude = np.abs(f)
        
        # OPTIMIZATION: Use FFT-based correlation (much faster)
        # autocorr = ifft(fft(x) * conj(fft(x)))
        autocorr = np.fft.irfft2(f * np.conj(f), s=image_small.shape)
        autocorr = np.fft.fftshift(autocorr)
        autocorr = autocorr / np.max(autocorr)
        
        h_small, w_small = autocorr.shape
        cy, cx = h_small // 2, w_small // 2
        
        # Mask center
        y, x = np.ogrid[:h_small, :w_small]
        center_mask = np.sqrt((x - cx)**2 + (y - cy)**2) < min(h_small, w_small) * 0.1
        autocorr_masked = autocorr.copy()
        autocorr_masked[center_mask] = 0
        
        # Find peaks
        peak_threshold = 0.3
        peaks = autocorr_masked > peak_threshold
        num_peaks = np.sum(peaks)
        
        max_secondary_peak = np.max(autocorr_masked) if num_peaks > 0 else 0.0
        repetition_score = min(1.0, max_secondary_peak * 2.0 + num_peaks / 100.0)
        
        is_suspicious = repetition_score > 0.35
        
        return {
            "repetition_score": float(repetition_score),
            "num_peaks": int(num_peaks),
            "max_secondary_peak": float(max_secondary_peak),
            "is_suspicious": bool(is_suspicious),
            "interpretation": "High score (>0.35) suggests texture repetition"
        }
    
    def _analyze_noise_residuals(self, image: np.ndarray) -> Dict:
        """Optimized noise residual analysis."""
        # OPTIMIZATION: Faster gaussian filter
        denoised = cv2.GaussianBlur(image, (5, 5), 1.5)
        residual = image.astype(float) - denoised.astype(float)
        
        # Downsample residual for FFT
        if max(residual.shape) > 512:
            scale = 512 / max(residual.shape)
            new_shape = (int(residual.shape[1] * scale), int(residual.shape[0] * scale))
            residual_small = cv2.resize(residual, new_shape)
        else:
            residual_small = residual
        
        # Analyze residual spectrum
        f = np.fft.rfft2(residual_small)
        magnitude = np.abs(f)
        
        # Compute entropy (faster)
        mag_flat = magnitude.flatten()
        mag_flat = mag_flat / (np.sum(mag_flat) + 1e-10)
        
        hist, _ = np.histogram(mag_flat, bins=50, density=True)  # Fewer bins
        hist = hist[hist > 0]
        
        entropy = -np.sum(hist * np.log2(hist + 1e-10))
        max_entropy = np.log2(50)
        normalized_entropy = entropy / max_entropy
        
        # Quick autocorrelation check (simplified)
        autocorr_fft = np.fft.irfft2(f * np.conj(f), s=residual_small.shape)
        autocorr_fft = np.fft.fftshift(autocorr_fft)
        autocorr_fft = autocorr_fft / np.max(np.abs(autocorr_fft))
        
        h, w = autocorr_fft.shape
        cy, cx = h // 2, w // 2
        
        # Sample ring
        y, x = np.ogrid[:h, :w]
        ring_mask = (np.sqrt((x - cx)**2 + (y - cy)**2) > 5) & \
                    (np.sqrt((x - cx)**2 + (y - cy)**2) < 20)
        
        if np.any(ring_mask):
            ring_autocorr = np.mean(np.abs(autocorr_fft[ring_mask]))
        else:
            ring_autocorr = 0.0
        
        structure_score = min(1.0, ring_autocorr * 3.0)
        entropy_score = 1.0 - normalized_entropy
        residual_score = (structure_score + entropy_score) / 2.0
        
        is_suspicious = residual_score > 0.5
        
        return {
            "residual_score": float(residual_score),
            "normalized_entropy": float(normalized_entropy),
            "structure_score": float(structure_score),
            "is_suspicious": bool(is_suspicious),
            "interpretation": "High score (>0.5) indicates denoising artifacts"
        }
    
    def _analyze_cross_channel_correlation(self, image_rgb: np.ndarray) -> Dict:
        """Optimized cross-channel analysis."""
        if len(image_rgb.shape) != 3 or image_rgb.shape[2] != 3:
            return {
                "correlation_score": 0.0,
                "is_suspicious": False,
                "interpretation": "Not RGB"
            }
        
        # OPTIMIZATION: Downsample for FFT
        if max(image_rgb.shape[:2]) > 512:
            scale = 512 / max(image_rgb.shape[:2])
            new_shape = (int(image_rgb.shape[1] * scale), int(image_rgb.shape[0] * scale))
            img_small = cv2.resize(image_rgb, new_shape)
        else:
            img_small = image_rgb
        
        # Compute FFT for each channel (use rfft2)
        r_fft = np.fft.rfft2(img_small[:,:,0])
        g_fft = np.fft.rfft2(img_small[:,:,1])
        b_fft = np.fft.rfft2(img_small[:,:,2])
        
        r_mag = np.abs(r_fft)
        g_mag = np.abs(g_fft)
        b_mag = np.abs(b_fft)
        
        # Focus on mid-frequencies
        h, w = r_mag.shape
        cy, cx = h // 2, w // 2
        y, x = np.ogrid[:h, :w]
        r_dist = np.sqrt((x - cx)**2 + (y - cy)**2)
        
        mid_freq_mask = (r_dist > min(h,w) * 0.1) & (r_dist < min(h,w) * 0.4)
        
        if not np.any(mid_freq_mask):
            return {
                "correlation_score": 0.0,
                "avg_correlation": 1.0,
                "is_suspicious": False,
                "interpretation": "Unable to analyze"
            }
        
        # Compute correlations
        r_flat = r_mag[mid_freq_mask].flatten()
        g_flat = g_mag[mid_freq_mask].flatten()
        b_flat = b_mag[mid_freq_mask].flatten()
        
        rg_corr = np.corrcoef(r_flat, g_flat)[0,1]
        rb_corr = np.corrcoef(r_flat, b_flat)[0,1]
        gb_corr = np.corrcoef(g_flat, b_flat)[0,1]
        
        avg_corr = (rg_corr + rb_corr + gb_corr) / 3.0
        correlation_score = 1.0 - avg_corr
        is_suspicious = avg_corr < 0.75
        
        return {
            "correlation_score": float(correlation_score),
            "avg_correlation": float(avg_corr),
            "rg_correlation": float(rg_corr),
            "rb_correlation": float(rb_corr),
            "gb_correlation": float(gb_corr),
            "is_suspicious": bool(is_suspicious),
            "interpretation": "Low correlation (<0.75) suggests independent synthesis"
        }
    
    def _multi_scale_spectral_analysis(self, image: np.ndarray) -> Dict:
        """Optimized multi-scale analysis."""
        scales = [1.0, 0.5, 0.25]
        scale_features = []
        
        for scale in scales:
            # Resize
            if scale < 1.0:
                new_h = int(image.shape[0] * scale)
                new_w = int(image.shape[1] * scale)
                # Ensure minimum size
                if new_h < 32 or new_w < 32:
                    continue
                scaled = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            else:
                scaled = image
            
            # Further downsample if needed
            if max(scaled.shape) > 256:
                ds_scale = 256 / max(scaled.shape)
                scaled = cv2.resize(scaled, 
                                   (int(scaled.shape[1] * ds_scale), 
                                    int(scaled.shape[0] * ds_scale)))
            
            # Compute FFT
            f = np.fft.rfft2(scaled)
            magnitude = np.abs(f)
            
            h, w = magnitude.shape
            cy, cx = h // 2, w // 2
            y, x = np.ogrid[:h, :w]
            r = np.sqrt((x - cx)**2 + (y - cy)**2)
            
            # Quick radial average
            max_radius = min(h, w) // 2
            radii = np.arange(1, max_radius, 2)  # Sample every 2nd radius
            radial_psd = []
            
            for radius in radii:
                mask = (r >= radius) & (r < radius + 2)
                if np.any(mask):
                    radial_psd.append(np.mean(magnitude[mask] ** 2))
            
            if len(radial_psd) > 5:
                radial_psd = np.array(radial_psd)
                log_r = np.log(radii[:len(radial_psd)])
                log_psd = np.log(radial_psd + 1e-10)
                
                slope = np.polyfit(log_r, log_psd, 1)[0]
                alpha = -slope
                
                # R-squared
                predicted = slope * log_r + np.polyfit(log_r, log_psd, 1)[1]
                r_squared = 1 - np.sum((log_psd - predicted)**2) / np.sum((log_psd - np.mean(log_psd))**2)
                
                scale_features.append({
                    'scale': scale,
                    'alpha': alpha,
                    'r_squared': r_squared
                })
        
        if len(scale_features) < 2:
            return {
                "multi_scale_score": 0.0,
                "is_suspicious": False,
                "interpretation": "Insufficient scales"
            }
        
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
        """Compute final verdict from all tests."""
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
        
        if confidence >= 0.6:
            verdict = "LIKELY_DIFFUSION_MODEL"
            explanation = f"Multiple diffusion signals ({suspicious_count}/{total_checks})"
        elif confidence >= 0.4:
            verdict = "SUSPICIOUS"
            explanation = f"Some anomalies detected ({suspicious_count}/{total_checks})"
        else:
            verdict = "LIKELY_REAL"
            explanation = f"Few anomalies ({suspicious_count}/{total_checks})"
        
        return {
            "verdict": verdict,
            "confidence": float(confidence),
            "suspicious_signals": suspicious_count,
            "total_checks": total_checks,
            "detailed_checks": checks,
            "explanation": explanation,
            "disclaimer": "Optimized detector for diffusion models. Not 100% accurate."
        }
    
    def _generate_visualizations(self, gray_image, rgb_image, image_path, output_dir):
        """Generate visualization (simplified for speed)."""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        fig = plt.figure(figsize=(16, 10))
        
        # 1. Original
        plt.subplot(2, 4, 1)
        plt.imshow(cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB))
        plt.title('Original Image', fontsize=11, fontweight='bold')
        plt.axis('off')
        
        # 2-6. Test results
        tests = [
            ("Local Freq", "local_frequency_inconsistency"),
            ("Texture Rep", "texture_repetition"),
            ("Noise Resid", "noise_residual_analysis"),
            ("Cross-Chan", "cross_channel_correlation"),
            ("Multi-Scale", "multi_scale_spectral")
        ]
        
        for idx, (title, key) in enumerate(tests, 2):
            plt.subplot(2, 4, idx)
            result = self.results[key]
            
            # Get main score
            score_key = [k for k in result.keys() if 'score' in k][0]
            score = result[score_key]
            status = "⚠️  SUSPICIOUS" if result["is_suspicious"] else "✓ PASS"
            time_taken = result.get('processing_time', 0)
            
            text = f'{title}\n\nScore: {score:.3f}\n'
            text += f'Status: {status}\n'
            text += f'Time: {time_taken:.2f}s'
            
            color = 'lightyellow' if result["is_suspicious"] else 'lightgreen'
            plt.text(0.5, 0.5, text, ha='center', va='center', fontsize=10,
                    bbox=dict(boxstyle='round', facecolor=color))
            plt.title(f'Test {idx-1}', fontsize=11, fontweight='bold')
            plt.axis('off')
        
        # 7. Verdict
        plt.subplot(2, 4, 7)
        v = self.results["verdict"]
        verdict_text = f'{v["verdict"]}\n\n'
        verdict_text += f'Confidence: {v["confidence"]:.1%}\n'
        verdict_text += f'Signals: {v["suspicious_signals"]}/{v["total_checks"]}\n\n'
        verdict_text += f'Total: {self.results["total_processing_time"]:.2f}s'
        
        color = 'lightcoral' if v["verdict"] == "LIKELY_DIFFUSION_MODEL" else \
                'lightyellow' if v["verdict"] == "SUSPICIOUS" else 'lightgreen'
        
        plt.text(0.5, 0.5, verdict_text, ha='center', va='center', 
                fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor=color, alpha=0.8))
        plt.title('Verdict', fontsize=11, fontweight='bold')
        plt.axis('off')
        
        # 8. Info
        plt.subplot(2, 4, 8)
        info = f'Image Size:\n{self.results["original_shape"][:2]}\n\n'
        info += f'Processed:\n{self.results["processed_shape"][:2]}\n\n'
        info += f'Resize: {self.results["resize_factor"]:.2f}x'
        
        plt.text(0.5, 0.5, info, ha='center', va='center', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='lightgray'))
        plt.title('Processing Info', fontsize=11, fontweight='bold')
        plt.axis('off')
        
        plt.tight_layout()
        
        output_path = os.path.join(output_dir, 'diffusion_analysis.png')
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        if self.verbose:
            print(f"✓ Saved: {output_path}")


def main():
    """Command line interface."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python optimized_diffusion_detector.py <image_path> [output_dir] [max_size]")
        print("\nExample:")
        print("  python optimized_diffusion_detector.py image.jpg results/ 1024")
        print("\nOptions:")
        print("  max_size: Maximum dimension (default: 1024, larger images are downsampled)")
        sys.exit(1)
    
    image_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "./diffusion_analysis"
    max_dimension = int(sys.argv[3]) if len(sys.argv) > 3 else 1024
    
    print("="*60)
    print("OPTIMIZED DIFFUSION MODEL DETECTOR")
    print("="*60)
    
    detector = OptimizedDiffusionDetector(max_dimension=max_dimension, verbose=True)
    results = detector.analyze_image(image_path, output_dir)
    
    if "error" in results:
        print(f"\n❌ Error: {results['error']}")
        sys.exit(1)
    
    # Print summary
    print(f"\n{'='*60}")
    print("FINAL VERDICT")
    print(f"{'='*60}")
    v = results["verdict"]
    print(f"\n   {v['verdict']}")
    print(f"   Confidence: {v['confidence']:.1%}")
    print(f"   Signals: {v['suspicious_signals']}/{v['total_checks']}")
    print(f"   Total time: {results['total_processing_time']:.2f}s")
    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    main()
