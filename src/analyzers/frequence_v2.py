import cv2
import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import signal, stats
from typing import Dict, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')


class FrequencyDeepfakeDetector:
    """
    Advanced frequency-domain analysis for deepfake detection.

    This detector uses multiple frequency-domain techniques:
    1. Checkerboard artifact detection (from upsampling in GANs)
    2. Power spectral density analysis (1/f distribution)
    3. Azimuthal average analysis
    4. High-frequency artifact detection
    5. Phase coherence analysis
    """

    def __init__(self):
        self.results = {}

    def analyze_image(self, image_path: str, output_dir: Optional[str] = None) -> Dict:
        """
        Perform comprehensive frequency analysis on an image.

        Args:
            image_path: Path to the image file
            output_dir: Directory to save visualization plots (optional)

        Returns:
            Dictionary containing all analysis results and verdict
        """
        try:
            # Read image
            img = cv2.imread(image_path)
            if img is None:
                return {"error": "Could not read image"}

            # Convert to grayscale for frequency analysis
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Perform all analyses
            self.results = {
                "image_shape": gray.shape,
                "checkerboard_score": self._detect_checkerboard_artifacts(gray),
                "power_law_analysis": self._analyze_power_law(gray),
                "azimuthal_analysis": self._azimuthal_average_analysis(gray),
                "high_freq_artifacts": self._detect_high_freq_artifacts(gray),
                "phase_analysis": self._analyze_phase_coherence(gray),
            }

            # Compute final verdict
            self.results["verdict"] = self._compute_verdict()

            # Generate visualizations if requested
            if output_dir:
                self._generate_visualizations(gray, image_path, output_dir)

            return self.results

        except Exception as e:
            return {"error": str(e)}

    def _compute_fft(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute 2D FFT and return magnitude and phase."""
        f = np.fft.fft2(image)
        fshift = np.fft.fftshift(f)

        magnitude = np.abs(fshift)
        phase = np.angle(fshift)

        return magnitude, phase

    def _detect_checkerboard_artifacts(self, image: np.ndarray) -> Dict:
        """
        Detect checkerboard patterns in frequency domain.

        GAN upsampling layers (especially transposed convolution) create
        checkerboard artifacts that appear as periodic peaks in frequency domain.

        Returns:
            Dictionary with checkerboard detection metrics
        """
        magnitude, _ = self._compute_fft(image)
        h, w = magnitude.shape
        cy, cx = h // 2, w // 2

        # Create radial frequency bins
        y, x = np.ogrid[:h, :w]
        r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

        # Compute radial profile (azimuthal average)
        max_radius = min(h, w) // 2
        radial_bins = np.arange(0, max_radius, 1)
        radial_profile = []

        for i in range(len(radial_bins) - 1):
            mask = (r >= radial_bins[i]) & (r < radial_bins[i + 1])
            if np.any(mask):
                radial_profile.append(np.mean(magnitude[mask]))
            else:
                radial_profile.append(0)

        radial_profile = np.array(radial_profile)

        # Normalize
        if np.max(radial_profile) > 0:
            radial_profile = radial_profile / np.max(radial_profile)

        # Detect periodic peaks (checkerboard signature)
        # Use autocorrelation to find periodic patterns
        if len(radial_profile) > 10:
            # Remove DC component and low frequencies for peak detection
            high_pass = radial_profile[5:]

            if len(high_pass) > 20:
                # Find peaks
                peaks, properties = signal.find_peaks(high_pass,
                                                      prominence=0.01,
                                                      distance=3)

                # Check for periodic peaks (sign of upsampling artifacts)
                num_peaks = len(peaks)

                # Calculate peak periodicity
                if num_peaks > 1:
                    peak_distances = np.diff(peaks)
                    periodicity_score = 1.0 / (1.0 + np.std(peak_distances))
                else:
                    periodicity_score = 0.0

                # Checkerboard score: combination of number of peaks and periodicity
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
            "radial_profile": radial_profile.tolist()[:100],  # Limit size
            "interpretation": "High score (>0.3) suggests upsampling artifacts typical of GANs"
        }

    def _analyze_power_law(self, image: np.ndarray) -> Dict:
        """
        Analyze power spectral density (PSD) for 1/f distribution.

        Natural images follow 1/f^α power law (α ≈ 2).
        Deviations suggest manipulation.

        Returns:
            Dictionary with power law analysis results
        """
        magnitude, _ = self._compute_fft(image)
        h, w = magnitude.shape
        cy, cx = h // 2, w // 2

        # Create radial frequency bins
        y, x = np.ogrid[:h, :w]
        r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

        # Compute power spectral density
        psd = magnitude ** 2

        # Radial average of PSD
        max_radius = min(h, w) // 2
        radial_bins = np.arange(1, max_radius, 1)  # Start from 1 to avoid DC
        radial_psd = []

        for i in range(len(radial_bins) - 1):
            mask = (r >= radial_bins[i]) & (r < radial_bins[i + 1])
            if np.any(mask):
                radial_psd.append(np.mean(psd[mask]))
            else:
                radial_psd.append(0)

        radial_psd = np.array(radial_psd)
        frequencies = radial_bins[:-1]

        # Fit power law: PSD ~ 1/f^α
        # Take log: log(PSD) ~ -α * log(f)
        valid_idx = (radial_psd > 0) & (frequencies > 0)

        if np.sum(valid_idx) > 10:
            log_freq = np.log(frequencies[valid_idx])
            log_psd = np.log(radial_psd[valid_idx])

            # Linear regression in log-log space
            slope, intercept, r_value, p_value, std_err = stats.linregress(log_freq, log_psd)

            # α = -slope (we want positive exponent)
            alpha = -slope
            r_squared = r_value ** 2

            # Natural images: α ≈ 1.5 to 2.5
            # GAN images often deviate from this
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
            "expected_alpha_range": [1.5, 2.5],
            "interpretation": "Natural images have α ≈ 2.0. Deviation > 0.7 suggests manipulation"
        }

    def _azimuthal_average_analysis(self, image: np.ndarray) -> Dict:
        """
        Analyze azimuthal (angular) variance in frequency spectrum.

        Natural images have roughly isotropic frequency distribution.
        GAN artifacts can create anisotropic patterns.

        Returns:
            Dictionary with azimuthal analysis results
        """
        magnitude, _ = self._compute_fft(image)
        h, w = magnitude.shape
        cy, cx = h // 2, w // 2

        # Create polar coordinates
        y, x = np.ogrid[:h, :w]
        r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        theta = np.arctan2(y - cy, x - cx)

        # Analyze specific frequency band (mid-frequencies)
        r_min, r_max = min(h, w) // 8, min(h, w) // 4
        band_mask = (r >= r_min) & (r < r_max)

        if not np.any(band_mask):
            return {
                "anisotropy_score": 0.0,
                "is_suspicious": False,
                "interpretation": "Insufficient frequency data"
            }

        # Compute angular profile
        num_angles = 36  # 10-degree bins
        angle_bins = np.linspace(-np.pi, np.pi, num_angles + 1)
        angular_profile = []

        for i in range(num_angles):
            angle_mask = (theta >= angle_bins[i]) & (theta < angle_bins[i + 1]) & band_mask
            if np.any(angle_mask):
                angular_profile.append(np.mean(magnitude[angle_mask]))
            else:
                angular_profile.append(0)

        angular_profile = np.array(angular_profile)

        # Compute anisotropy: variance in angular distribution
        if np.mean(angular_profile) > 0:
            normalized_profile = angular_profile / np.mean(angular_profile)
            anisotropy_score = np.std(normalized_profile)
        else:
            anisotropy_score = 0.0

        # High anisotropy suggests artifacts
        is_suspicious = anisotropy_score > 0.25

        return {
            "anisotropy_score": float(anisotropy_score),
            "is_suspicious": bool(is_suspicious),
            "angular_profile": angular_profile.tolist(),
            "interpretation": "High anisotropy (>0.25) suggests directional artifacts"
        }

    def _detect_high_freq_artifacts(self, image: np.ndarray) -> Dict:
        """
        Detect anomalous high-frequency content.

        Some GANs produce either too little or too much high-frequency content.

        Returns:
            Dictionary with high-frequency analysis
        """
        magnitude, _ = self._compute_fft(image)
        h, w = magnitude.shape
        cy, cx = h // 2, w // 2

        # Create masks for different frequency bands
        y, x = np.ogrid[:h, :w]
        r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

        max_radius = min(h, w) // 2

        # Low frequencies (0-25%)
        low_mask = r < max_radius * 0.25
        # Mid frequencies (25-50%)
        mid_mask = (r >= max_radius * 0.25) & (r < max_radius * 0.5)
        # High frequencies (50-75%)
        high_mask = (r >= max_radius * 0.5) & (r < max_radius * 0.75)

        # Compute energy in each band
        low_energy = np.mean(magnitude[low_mask]) if np.any(low_mask) else 0
        mid_energy = np.mean(magnitude[mid_mask]) if np.any(mid_mask) else 0
        high_energy = np.mean(magnitude[high_mask]) if np.any(high_mask) else 0

        # Compute ratios
        if low_energy > 0:
            high_to_low_ratio = high_energy / low_energy
            mid_to_low_ratio = mid_energy / low_energy
        else:
            high_to_low_ratio = 0
            mid_to_low_ratio = 0

        # Natural images: smooth fall-off
        # Expected ratios (approximate): mid/low ≈ 0.1-0.3, high/low ≈ 0.01-0.1

        # Check for anomalies
        too_much_high_freq = high_to_low_ratio > 0.15  # Too much high-freq (JPEG, artifacts)
        too_little_high_freq = high_to_low_ratio < 0.01  # Too smooth (over-processed GAN)

        is_suspicious = too_much_high_freq or too_little_high_freq

        return {
            "low_freq_energy": float(low_energy),
            "mid_freq_energy": float(mid_energy),
            "high_freq_energy": float(high_energy),
            "high_to_low_ratio": float(high_to_low_ratio),
            "mid_to_low_ratio": float(mid_to_low_ratio),
            "too_much_high_freq": bool(too_much_high_freq),
            "too_little_high_freq": bool(too_little_high_freq),
            "is_suspicious": bool(is_suspicious),
            "interpretation": "Natural range: high/low ≈ 0.01-0.15. Deviations suggest manipulation"
        }

    def _analyze_phase_coherence(self, image: np.ndarray) -> Dict:
        """
        Analyze phase information in frequency domain.

        Natural images have specific phase statistics.
        Some GAN artifacts appear in phase rather than magnitude.

        Returns:
            Dictionary with phase analysis results
        """
        _, phase = self._compute_fft(image)
        h, w = phase.shape
        cy, cx = h // 2, w // 2

        # Analyze phase distribution
        # Natural images have relatively uniform phase distribution

        # Compute phase histogram
        phase_hist, _ = np.histogram(phase.flatten(), bins=50, range=(-np.pi, np.pi))
        phase_hist = phase_hist / np.sum(phase_hist)  # Normalize

        # Compute entropy of phase distribution
        # High entropy = more uniform = more natural
        phase_hist_nonzero = phase_hist[phase_hist > 0]
        phase_entropy = -np.sum(phase_hist_nonzero * np.log2(phase_hist_nonzero))

        # Maximum entropy for 50 bins
        max_entropy = np.log2(50)
        normalized_entropy = phase_entropy / max_entropy

        # Low entropy suggests non-natural phase distribution
        is_suspicious = normalized_entropy < 0.85

        # Compute phase coherence in different frequency bands
        y, x = np.ogrid[:h, :w]
        r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        max_radius = min(h, w) // 2

        # High frequency phase variance
        high_freq_mask = (r >= max_radius * 0.5) & (r < max_radius * 0.75)
        if np.any(high_freq_mask):
            high_freq_phase_var = np.var(phase[high_freq_mask])
        else:
            high_freq_phase_var = 0.0

        return {
            "phase_entropy": float(phase_entropy),
            "normalized_entropy": float(normalized_entropy),
            "high_freq_phase_variance": float(high_freq_phase_var),
            "is_suspicious": bool(is_suspicious),
            "interpretation": "Low entropy (<0.85) suggests non-natural phase distribution"
        }

    def _compute_verdict(self) -> Dict:
        """
        Combine all analyses to produce final verdict.

        Returns:
            Dictionary with overall assessment
        """
        # Count suspicious signals
        suspicious_count = 0
        total_checks = 5

        checks = {
            "checkerboard_artifacts": self.results["checkerboard_score"]["is_suspicious"],
            "power_law_deviation": self.results["power_law_analysis"]["is_suspicious"],
            "azimuthal_anisotropy": self.results["azimuthal_analysis"]["is_suspicious"],
            "high_freq_anomaly": self.results["high_freq_artifacts"]["is_suspicious"],
            "phase_anomaly": self.results["phase_analysis"]["is_suspicious"]
        }

        suspicious_count = sum(checks.values())

        # Compute confidence score (0-1)
        confidence = suspicious_count / total_checks

        # Determine verdict
        if confidence >= 0.6:
            verdict = "LIKELY_FAKE"
            explanation = f"Multiple suspicious signals detected ({suspicious_count}/{total_checks})"
        elif confidence >= 0.4:
            verdict = "SUSPICIOUS"
            explanation = f"Some anomalies detected ({suspicious_count}/{total_checks}). Further analysis recommended"
        else:
            verdict = "LIKELY_REAL"
            explanation = f"Few anomalies detected ({suspicious_count}/{total_checks}). Appears natural"

        return {
            "verdict": verdict,
            "confidence": float(confidence),
            "suspicious_signals": suspicious_count,
            "total_checks": total_checks,
            "detailed_checks": checks,
            "explanation": explanation,
            "disclaimer": "This is a heuristic analysis. Modern GANs can fool frequency-domain detectors. Use ML-based approaches for production systems."
        }

    def _generate_visualizations(self, image: np.ndarray, image_path: str, output_dir: str):
        """Generate comprehensive visualization plots."""
        import os
        os.makedirs(output_dir, exist_ok=True)

        magnitude, phase = self._compute_fft(image)

        # Create log magnitude spectrum for visualization
        magnitude_db = 20 * np.log10(magnitude + 1)

        # Create comprehensive figure
        fig = plt.figure(figsize=(20, 12))

        # 1. Original image
        plt.subplot(3, 4, 1)
        plt.imshow(image, cmap='gray')
        plt.title('Original Image', fontsize=12, fontweight='bold')
        plt.axis('off')

        # 2. Magnitude spectrum
        plt.subplot(3, 4, 2)
        plt.imshow(magnitude_db, cmap='hot')
        plt.title('Magnitude Spectrum (dB)', fontsize=12, fontweight='bold')
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.axis('off')

        # 3. Phase spectrum
        plt.subplot(3, 4, 3)
        plt.imshow(phase, cmap='twilight')
        plt.title('Phase Spectrum', fontsize=12, fontweight='bold')
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.axis('off')

        # 4. Radial profile (checkerboard detection)
        plt.subplot(3, 4, 4)
        radial_profile = self.results["checkerboard_score"]["radial_profile"]
        plt.plot(radial_profile, linewidth=2)
        plt.title(f'Radial Profile\nCheckerboard Score: {self.results["checkerboard_score"]["score"]:.3f}',
                  fontsize=12, fontweight='bold')
        plt.xlabel('Frequency (pixels)')
        plt.ylabel('Normalized Magnitude')
        plt.grid(True, alpha=0.3)

        # 5. Power law fit
        plt.subplot(3, 4, 5)
        alpha = self.results["power_law_analysis"]["alpha"]
        r_sq = self.results["power_law_analysis"]["r_squared"]
        plt.text(0.5, 0.5, f'Power Law Analysis\n\nα = {alpha:.3f}\n(Expected: ~2.0)\n\nR² = {r_sq:.3f}',
                 ha='center', va='center', fontsize=14,
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        plt.title('1/f^α Analysis', fontsize=12, fontweight='bold')
        plt.axis('off')

        # 6. Angular profile
        plt.subplot(3, 4, 6)
        angular_profile = self.results["azimuthal_analysis"]["angular_profile"]
        angles = np.linspace(0, 360, len(angular_profile))
        plt.plot(angles, angular_profile, linewidth=2)
        plt.title(f'Angular Profile\nAnisotropy: {self.results["azimuthal_analysis"]["anisotropy_score"]:.3f}',
                  fontsize=12, fontweight='bold')
        plt.xlabel('Angle (degrees)')
        plt.ylabel('Magnitude')
        plt.grid(True, alpha=0.3)

        # 7. Frequency band energies
        plt.subplot(3, 4, 7)
        energies = [
            self.results["high_freq_artifacts"]["low_freq_energy"],
            self.results["high_freq_artifacts"]["mid_freq_energy"],
            self.results["high_freq_artifacts"]["high_freq_energy"]
        ]
        bands = ['Low\n(0-25%)', 'Mid\n(25-50%)', 'High\n(50-75%)']
        bars = plt.bar(bands, energies, color=['green', 'orange', 'red'], alpha=0.7)
        plt.title('Frequency Band Energy', fontsize=12, fontweight='bold')
        plt.ylabel('Average Magnitude')
        plt.grid(True, alpha=0.3, axis='y')

        # 8. Phase entropy
        plt.subplot(3, 4, 8)
        entropy = self.results["phase_analysis"]["normalized_entropy"]
        plt.text(0.5, 0.5, f'Phase Entropy\n\n{entropy:.3f}\n(Expected: >0.85)',
                 ha='center', va='center', fontsize=14,
                 bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        plt.title('Phase Analysis', fontsize=12, fontweight='bold')
        plt.axis('off')

        # 9-12. Summary and verdict
        verdict_subplot = plt.subplot(3, 4, 9)
        verdict = self.results["verdict"]
        verdict_text = f'VERDICT: {verdict["verdict"]}\n\n'
        verdict_text += f'Confidence: {verdict["confidence"]:.1%}\n\n'
        verdict_text += f'Suspicious Signals: {verdict["suspicious_signals"]}/{verdict["total_checks"]}\n\n'

        # Color based on verdict
        if verdict["verdict"] == "LIKELY_FAKE":
            color = 'lightcoral'
        elif verdict["verdict"] == "SUSPICIOUS":
            color = 'lightyellow'
        else:
            color = 'lightgreen'

        plt.text(0.5, 0.5, verdict_text,
                 ha='center', va='center', fontsize=14, fontweight='bold',
                 bbox=dict(boxstyle='round', facecolor=color, alpha=0.8))
        plt.title('Overall Assessment', fontsize=12, fontweight='bold')
        plt.axis('off')

        # 10. Detailed checks
        plt.subplot(3, 4, 10)
        checks = verdict["detailed_checks"]
        check_text = "Detailed Checks:\n\n"
        for check_name, is_suspicious in checks.items():
            status = "⚠️ FAIL" if is_suspicious else "✓ PASS"
            check_text += f"{status} {check_name.replace('_', ' ').title()}\n"

        plt.text(0.05, 0.95, check_text,
                 ha='left', va='top', fontsize=10, family='monospace',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        plt.title('Check Details', fontsize=12, fontweight='bold')
        plt.axis('off')

        # 11-12. Explanation
        explanation_text = verdict["explanation"] + "\n\n" + verdict["disclaimer"]
        plt.subplot(3, 4, 11)
        plt.text(0.05, 0.95, explanation_text,
                 ha='left', va='top', fontsize=9, wrap=True,
                 bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
        plt.title('Explanation', fontsize=12, fontweight='bold')
        plt.axis('off')

        plt.subplot(3, 4, 12)
        disclaimer = ("IMPORTANT NOTES:\n\n"
                      "• Modern GANs (2020+) often pass these tests\n"
                      "• Requires ML-based detection for production\n"
                      "• JPEG compression affects results\n"
                      "• Image size and quality matter\n"
                      "• Use as one signal among many")
        plt.text(0.05, 0.95, disclaimer,
                 ha='left', va='top', fontsize=9, wrap=True,
                 bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
        plt.title('Limitations', fontsize=12, fontweight='bold')
        plt.axis('off')

        plt.tight_layout()

        # Save figure
        output_path = os.path.join(output_dir, 'frequency_analysis.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"\n✓ Visualization saved to: {output_path}")


def main():
    """Example usage of the FrequencyDeepfakeDetector."""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python frequency_deepfake_detector.py <image_path> [output_dir]")
        print("\nExample:")
        print("  python frequency_deepfake_detector.py test_image.jpg ./results")
        sys.exit(1)

    image_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "./frequency_analysis_output"

    print("=" * 80)
    print("FREQUENCY-DOMAIN DEEPFAKE DETECTOR")
    print("=" * 80)
    print(f"\nAnalyzing: {image_path}")
    print(f"Output directory: {output_dir}\n")

    # Create detector and analyze
    detector = FrequencyDeepfakeDetector()
    results = detector.analyze_image(image_path, output_dir)

    if "error" in results:
        print(f"❌ Error: {results['error']}")
        sys.exit(1)

    # Print results
    print("\n" + "=" * 80)
    print("ANALYSIS RESULTS")
    print("=" * 80)

    print("\n1. CHECKERBOARD ARTIFACT DETECTION")
    print("-" * 40)
    cb = results["checkerboard_score"]
    print(f"   Score: {cb['score']:.3f}")
    print(f"   Peaks detected: {cb['num_peaks']}")
    print(f"   Periodicity: {cb['periodicity']:.3f}")
    print(f"   Status: {'⚠️  SUSPICIOUS' if cb['is_suspicious'] else '✓ PASS'}")
    print(f"   {cb['interpretation']}")

    print("\n2. POWER LAW ANALYSIS (1/f Distribution)")
    print("-" * 40)
    pl = results["power_law_analysis"]
    print(f"   Alpha (α): {pl['alpha']:.3f}")
    print(f"   Expected range: {pl['expected_alpha_range']}")
    print(f"   Deviation: {pl['alpha_deviation']:.3f}")
    print(f"   R-squared: {pl['r_squared']:.3f}")
    print(f"   Status: {'⚠️  SUSPICIOUS' if pl['is_suspicious'] else '✓ PASS'}")
    print(f"   {pl['interpretation']}")

    print("\n3. AZIMUTHAL ANALYSIS")
    print("-" * 40)
    az = results["azimuthal_analysis"]
    print(f"   Anisotropy score: {az['anisotropy_score']:.3f}")
    print(f"   Status: {'⚠️  SUSPICIOUS' if az['is_suspicious'] else '✓ PASS'}")
    print(f"   {az['interpretation']}")

    print("\n4. HIGH-FREQUENCY ARTIFACTS")
    print("-" * 40)
    hf = results["high_freq_artifacts"]
    print(f"   High/Low ratio: {hf['high_to_low_ratio']:.4f}")
    print(f"   Mid/Low ratio: {hf['mid_to_low_ratio']:.4f}")
    print(f"   Too much high-freq: {hf['too_much_high_freq']}")
    print(f"   Too little high-freq: {hf['too_little_high_freq']}")
    print(f"   Status: {'⚠️  SUSPICIOUS' if hf['is_suspicious'] else '✓ PASS'}")
    print(f"   {hf['interpretation']}")

    print("\n5. PHASE COHERENCE ANALYSIS")
    print("-" * 40)
    ph = results["phase_analysis"]
    print(f"   Normalized entropy: {ph['normalized_entropy']:.3f}")
    print(f"   High-freq phase variance: {ph['high_freq_phase_variance']:.3f}")
    print(f"   Status: {'⚠️  SUSPICIOUS' if ph['is_suspicious'] else '✓ PASS'}")
    print(f"   {ph['interpretation']}")

    print("\n" + "=" * 80)
    print("FINAL VERDICT")
    print("=" * 80)
    v = results["verdict"]
    print(f"\n   Verdict: {v['verdict']}")
    print(f"   Confidence: {v['confidence']:.1%}")
    print(f"   Suspicious signals: {v['suspicious_signals']}/{v['total_checks']}")
    print(f"\n   {v['explanation']}")
    print(f"\n   ⚠️  {v['disclaimer']}")

    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
