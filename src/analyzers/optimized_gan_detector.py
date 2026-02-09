"""
Optimized Frequency-Domain GAN Detector

PERFORMANCE OPTIMIZATIONS:
- Automatic image downsampling for large images  
- Efficient FFT operations (rfft2 instead of fft2)
- Progress bars with tqdm
- Reduced radial sampling
- Faster correlation methods
"""

import cv2
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import signal, stats
from typing import Dict, Tuple, Optional
import warnings
import time
from tqdm import tqdm
warnings.filterwarnings('ignore')


class OptimizedGANDetector:
    """
    Fast GAN detector with automatic resizing and progress tracking.
    """
    
    def __init__(self, max_dimension=1024, verbose=True):
        """
        Args:
            max_dimension: Maximum image dimension (larger images downsampled)
            verbose: Show progress bars and timing
        """
        self.max_dimension = max_dimension
        self.verbose = verbose
        self.results = {}
        
    def analyze_image(self, image_path: str, output_dir: Optional[str] = None) -> Dict:
        """Analyze image for GAN artifacts with progress tracking."""
        try:
            start_time = time.time()
            
            if self.verbose:
                print("📁 Loading image...")
            
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                return {"error": "Could not read image"}
            
            original_shape = img.shape
            
            # Resize if needed
            img, resize_factor = self._resize_if_needed(img)
            
            if self.verbose:
                print(f"✓ Image loaded: {original_shape} → {img.shape} (factor: {resize_factor:.2f})")
                print(f"\n{'='*60}")
                print("Running GAN Analysis (5 tests)")
                print(f"{'='*60}\n")
            
            # Compute FFT once (reuse for multiple analyses)
            magnitude, phase = self._compute_fft(img)
            
            # Run analyses with progress bar
            tests = [
                ("Checkerboard Artifacts", lambda: self._detect_checkerboard_artifacts(magnitude)),
                ("Power Law Analysis", lambda: self._analyze_power_law(magnitude)),
                ("Azimuthal Analysis", lambda: self._azimuthal_average_analysis(magnitude)),
                ("High-Freq Artifacts", lambda: self._detect_high_freq_artifacts(magnitude)),
                ("Phase Analysis", lambda: self._analyze_phase_coherence(phase))
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
                result = test_func()
                test_time = time.time() - test_start
                result['processing_time'] = test_time
                
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
                self._generate_visualizations(img, magnitude, phase, image_path, output_dir)
            
            return self.results
            
        except Exception as e:
            return {"error": str(e)}
    
    def _resize_if_needed(self, img: np.ndarray) -> Tuple[np.ndarray, float]:
        """Resize if larger than max_dimension."""
        h, w = img.shape
        max_dim = max(h, w)
        
        if max_dim <= self.max_dimension:
            return img, 1.0
        
        factor = self.max_dimension / max_dim
        new_h, new_w = int(h * factor), int(w * factor)
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        return resized, factor
    
    def _compute_fft(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute 2D FFT efficiently."""
        # Use rfft2 for real input (2x faster than fft2)
        f = np.fft.rfft2(image)
        fshift = np.fft.fftshift(f, axes=0)
        
        magnitude = np.abs(fshift)
        phase = np.angle(fshift)
        
        return magnitude, phase
    
    def _detect_checkerboard_artifacts(self, magnitude: np.ndarray) -> Dict:
        """Optimized checkerboard detection."""
        h, w = magnitude.shape
        cy, cx = h // 2, w // 2
        
        # Create radial coordinates
        y, x = np.ogrid[:h, :w]
        r = np.sqrt((x - cx)**2 + (y - cy)**2)
        
        # OPTIMIZATION: Sample fewer radial bins
        max_radius = min(h, w) // 2
        radial_bins = np.arange(0, max_radius, 2)  # Every 2nd pixel
        radial_profile = []
        
        for i in range(len(radial_bins) - 1):
            mask = (r >= radial_bins[i]) & (r < radial_bins[i + 1])
            if np.any(mask):
                radial_profile.append(np.mean(magnitude[mask]))
            else:
                radial_profile.append(0)
        
        radial_profile = np.array(radial_profile)
        
        if np.max(radial_profile) > 0:
            radial_profile = radial_profile / np.max(radial_profile)
        
        # Detect peaks (simplified)
        if len(radial_profile) > 20:
            high_pass = radial_profile[5:]
            
            if len(high_pass) > 20:
                # Quick peak detection
                peaks, _ = signal.find_peaks(high_pass, prominence=0.01, distance=3)
                num_peaks = len(peaks)
                
                if num_peaks > 1:
                    peak_distances = np.diff(peaks)
                    periodicity_score = 1.0 / (1.0 + np.std(peak_distances))
                else:
                    periodicity_score = 0.0
                
                checkerboard_score = min(1.0, (num_peaks / 10.0) * periodicity_score)
                is_suspicious = checkerboard_score > 0.3
            else:
                num_peaks = 0
                periodicity_score = 0.0
                checkerboard_score = 0.0
                is_suspicious = False
        else:
            num_peaks = 0
            periodicity_score = 0.0
            checkerboard_score = 0.0
            is_suspicious = False
        
        return {
            "score": float(checkerboard_score),
            "num_peaks": int(num_peaks),
            "periodicity": float(periodicity_score),
            "is_suspicious": bool(is_suspicious),
            "interpretation": "High score (>0.3) suggests upsampling artifacts"
        }
    
    def _analyze_power_law(self, magnitude: np.ndarray) -> Dict:
        """Optimized power law analysis."""
        h, w = magnitude.shape
        cy, cx = h // 2, w // 2
        
        # Create radial coordinates
        y, x = np.ogrid[:h, :w]
        r = np.sqrt((x - cx)**2 + (y - cy)**2)
        
        # Power spectral density
        psd = magnitude ** 2
        
        # OPTIMIZATION: Sample fewer radii
        max_radius = min(h, w) // 2
        radial_bins = np.arange(1, max_radius, 2)  # Every 2nd pixel
        radial_psd = []
        
        for i in range(len(radial_bins) - 1):
            mask = (r >= radial_bins[i]) & (r < radial_bins[i + 1])
            if np.any(mask):
                radial_psd.append(np.mean(psd[mask]))
            else:
                radial_psd.append(0)
        
        radial_psd = np.array(radial_psd)
        frequencies = radial_bins[:-1]
        
        # Fit power law
        valid_idx = (radial_psd > 0) & (frequencies > 0)
        
        if np.sum(valid_idx) > 10:
            log_freq = np.log(frequencies[valid_idx])
            log_psd = np.log(radial_psd[valid_idx])
            
            # Quick polyfit instead of linregress
            coeffs = np.polyfit(log_freq, log_psd, 1)
            slope = coeffs[0]
            
            alpha = -slope
            
            # Quick R-squared
            predicted = slope * log_freq + coeffs[1]
            r_squared = 1 - np.sum((log_psd - predicted)**2) / np.sum((log_psd - np.mean(log_psd))**2)
            
            alpha_deviation = abs(alpha - 2.0)
            is_suspicious = (alpha_deviation > 0.7) or (r_squared < 0.85)
        else:
            alpha = 0.0
            r_squared = 0.0
            alpha_deviation = 0.0
            is_suspicious = False
        
        return {
            "alpha": float(alpha),
            "r_squared": float(r_squared),
            "alpha_deviation": float(alpha_deviation),
            "is_suspicious": bool(is_suspicious),
            "interpretation": "Natural images: α ≈ 2.0. Deviation > 0.7 is suspicious"
        }
    
    def _azimuthal_average_analysis(self, magnitude: np.ndarray) -> Dict:
        """Optimized azimuthal analysis."""
        h, w = magnitude.shape
        cy, cx = h // 2, w // 2
        
        # Create coordinates
        y, x = np.ogrid[:h, :w]
        r = np.sqrt((x - cx)**2 + (y - cy)**2)
        theta = np.arctan2(y - cy, x - cx)
        
        # Mid-frequency band
        r_min, r_max = min(h, w) // 8, min(h, w) // 4
        band_mask = (r >= r_min) & (r < r_max)
        
        if not np.any(band_mask):
            return {
                "anisotropy_score": 0.0,
                "is_suspicious": False,
                "interpretation": "Insufficient data"
            }
        
        # OPTIMIZATION: Fewer angular bins
        num_angles = 18  # 20-degree bins (was 36)
        angle_bins = np.linspace(-np.pi, np.pi, num_angles + 1)
        angular_profile = []
        
        for i in range(num_angles):
            angle_mask = (theta >= angle_bins[i]) & (theta < angle_bins[i + 1]) & band_mask
            if np.any(angle_mask):
                angular_profile.append(np.mean(magnitude[angle_mask]))
            else:
                angular_profile.append(0)
        
        angular_profile = np.array(angular_profile)
        
        # Compute anisotropy
        if np.mean(angular_profile) > 0:
            normalized_profile = angular_profile / np.mean(angular_profile)
            anisotropy_score = np.std(normalized_profile)
        else:
            anisotropy_score = 0.0
        
        is_suspicious = anisotropy_score > 0.25
        
        return {
            "anisotropy_score": float(anisotropy_score),
            "is_suspicious": bool(is_suspicious),
            "interpretation": "High anisotropy (>0.25) suggests directional artifacts"
        }
    
    def _detect_high_freq_artifacts(self, magnitude: np.ndarray) -> Dict:
        """Optimized high-frequency artifact detection."""
        h, w = magnitude.shape
        cy, cx = h // 2, w // 2
        
        # Create radial coordinates
        y, x = np.ogrid[:h, :w]
        r = np.sqrt((x - cx)**2 + (y - cy)**2)
        
        max_radius = min(h, w) // 2
        
        # Frequency bands
        low_mask = r < max_radius * 0.25
        mid_mask = (r >= max_radius * 0.25) & (r < max_radius * 0.5)
        high_mask = (r >= max_radius * 0.5) & (r < max_radius * 0.75)
        
        # Compute energies
        low_energy = np.mean(magnitude[low_mask]) if np.any(low_mask) else 0
        mid_energy = np.mean(magnitude[mid_mask]) if np.any(mid_mask) else 0
        high_energy = np.mean(magnitude[high_mask]) if np.any(high_mask) else 0
        
        # Ratios
        if low_energy > 0:
            high_to_low_ratio = high_energy / low_energy
            mid_to_low_ratio = mid_energy / low_energy
        else:
            high_to_low_ratio = 0
            mid_to_low_ratio = 0
        
        too_much_high_freq = high_to_low_ratio > 0.15
        too_little_high_freq = high_to_low_ratio < 0.01
        is_suspicious = too_much_high_freq or too_little_high_freq
        
        return {
            "high_to_low_ratio": float(high_to_low_ratio),
            "mid_to_low_ratio": float(mid_to_low_ratio),
            "too_much_high_freq": bool(too_much_high_freq),
            "too_little_high_freq": bool(too_little_high_freq),
            "is_suspicious": bool(is_suspicious),
            "interpretation": "Natural range: 0.01-0.15. Deviations suspicious"
        }
    
    def _analyze_phase_coherence(self, phase: np.ndarray) -> Dict:
        """Optimized phase analysis."""
        # OPTIMIZATION: Sample phase instead of using all pixels
        sample_size = min(100000, phase.size)
        phase_sample = np.random.choice(phase.flatten(), size=sample_size, replace=False)
        
        # Compute histogram (fewer bins)
        phase_hist, _ = np.histogram(phase_sample, bins=50, range=(-np.pi, np.pi))
        phase_hist = phase_hist / np.sum(phase_hist)
        
        # Entropy
        phase_hist_nonzero = phase_hist[phase_hist > 0]
        phase_entropy = -np.sum(phase_hist_nonzero * np.log2(phase_hist_nonzero))
        
        max_entropy = np.log2(50)
        normalized_entropy = phase_entropy / max_entropy
        
        is_suspicious = normalized_entropy < 0.85
        
        return {
            "normalized_entropy": float(normalized_entropy),
            "is_suspicious": bool(is_suspicious),
            "interpretation": "Low entropy (<0.85) suggests non-natural phase"
        }
    
    def _compute_verdict(self) -> Dict:
        """Compute final verdict."""
        checks = {
            "checkerboard_artifacts": self.results["checkerboard_artifacts"]["is_suspicious"],
            "power_law_deviation": self.results["power_law_analysis"]["is_suspicious"],
            "azimuthal_anisotropy": self.results["azimuthal_analysis"]["is_suspicious"],
            "high_freq_anomaly": self.results["high_freq_artifacts"]["is_suspicious"],
            "phase_anomaly": self.results["phase_analysis"]["is_suspicious"]
        }
        
        suspicious_count = sum(checks.values())
        total_checks = len(checks)
        confidence = suspicious_count / total_checks
        
        if confidence >= 0.6:
            verdict = "LIKELY_GAN"
            explanation = f"Multiple GAN signals ({suspicious_count}/{total_checks})"
        elif confidence >= 0.4:
            verdict = "SUSPICIOUS"
            explanation = f"Some anomalies ({suspicious_count}/{total_checks})"
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
            "disclaimer": "Optimized for GANs. Use diffusion detector for modern AI art."
        }
    
    def _generate_visualizations(self, image, magnitude, phase, image_path, output_dir):
        """Generate visualizations (simplified for speed)."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Create log magnitude for visualization
        magnitude_db = 20 * np.log10(magnitude + 1)
        
        fig = plt.figure(figsize=(16, 10))
        
        # 1. Original
        plt.subplot(2, 4, 1)
        plt.imshow(image, cmap='gray')
        plt.title('Original Image', fontsize=11, fontweight='bold')
        plt.axis('off')
        
        # 2. Magnitude spectrum
        plt.subplot(2, 4, 2)
        plt.imshow(magnitude_db, cmap='hot')
        plt.title('Magnitude Spectrum', fontsize=11, fontweight='bold')
        plt.axis('off')
        
        # 3-6. Test results
        tests = [
            ("Checkerboard", "checkerboard_artifacts", "score"),
            ("Power Law", "power_law_analysis", "alpha"),
            ("Azimuthal", "azimuthal_analysis", "anisotropy_score"),
            ("High-Freq", "high_freq_artifacts", "high_to_low_ratio"),
            ("Phase", "phase_analysis", "normalized_entropy")
        ]
        
        for idx, (title, key, score_key) in enumerate(tests, 3):
            plt.subplot(2, 4, idx)
            result = self.results[key]
            
            score = result[score_key]
            status = "⚠️  SUSPICIOUS" if result["is_suspicious"] else "✓ PASS"
            time_taken = result.get('processing_time', 0)
            
            text = f'{title}\n\n{score_key}: {score:.3f}\n'
            text += f'Status: {status}\n'
            text += f'Time: {time_taken:.2f}s'
            
            color = 'lightyellow' if result["is_suspicious"] else 'lightgreen'
            plt.text(0.5, 0.5, text, ha='center', va='center', fontsize=9,
                    bbox=dict(boxstyle='round', facecolor=color))
            plt.title(f'Test {idx-2}', fontsize=11, fontweight='bold')
            plt.axis('off')
        
        # 7. Verdict
        plt.subplot(2, 4, 7)
        v = self.results["verdict"]
        verdict_text = f'{v["verdict"]}\n\n'
        verdict_text += f'Confidence: {v["confidence"]:.1%}\n'
        verdict_text += f'Signals: {v["suspicious_signals"]}/{v["total_checks"]}\n\n'
        verdict_text += f'Total: {self.results["total_processing_time"]:.2f}s'
        
        color = 'lightcoral' if v["verdict"] == "LIKELY_GAN" else \
                'lightyellow' if v["verdict"] == "SUSPICIOUS" else 'lightgreen'
        
        plt.text(0.5, 0.5, verdict_text, ha='center', va='center',
                fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor=color, alpha=0.8))
        plt.title('Verdict', fontsize=11, fontweight='bold')
        plt.axis('off')
        
        # 8. Info
        plt.subplot(2, 4, 8)
        info = f'Image Size:\n{self.results["original_shape"]}\n\n'
        info += f'Processed:\n{self.results["processed_shape"]}\n\n'
        info += f'Resize: {self.results["resize_factor"]:.2f}x'
        
        plt.text(0.5, 0.5, info, ha='center', va='center', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='lightgray'))
        plt.title('Processing Info', fontsize=11, fontweight='bold')
        plt.axis('off')
        
        plt.tight_layout()
        
        output_path = os.path.join(output_dir, 'gan_analysis.png')
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        if self.verbose:
            print(f"✓ Saved: {output_path}")


def main():
    """Command line interface."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python optimized_gan_detector.py <image_path> [output_dir] [max_size]")
        print("\nExample:")
        print("  python optimized_gan_detector.py image.jpg results/ 1024")
        sys.exit(1)
    
    image_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "./gan_analysis"
    max_dimension = int(sys.argv[3]) if len(sys.argv) > 3 else 1024
    
    print("="*60)
    print("OPTIMIZED GAN DETECTOR")
    print("="*60)
    
    detector = OptimizedGANDetector(max_dimension=max_dimension, verbose=True)
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
