"""
Unified Fast Deepfake Detector

Automatically runs BOTH GAN and Diffusion detectors,
then provides comprehensive verdict.

OPTIMIZED FOR SPEED:
- Auto-downsamples large images (1622x2285 → ~1024x1024)
- Progress bars for all operations
- Expected time: 5-15 seconds (vs 30+ minutes unoptimized)
"""

import sys
import os
import time

from src.analyzers.optimized_diffusion_detector import OptimizedDiffusionDetector
from src.analyzers.optimized_gan_detector import OptimizedGANDetector


class UnifiedDeepfakeDetector:
    """
    Runs both GAN and Diffusion detection, picks best result.
    """
    
    def __init__(self, max_dimension=1024, verbose=True):
        self.max_dimension = max_dimension
        self.verbose = verbose
        
    def analyze_image(self, image_path: str, output_dir: str = None):
        """
        Run both detectors and combine results.
        """
        if output_dir is None:
            output_dir = "./unified_analysis"
        
        os.makedirs(output_dir, exist_ok=True)
        
        if self.verbose:
            print("\n" + "="*70)
            print("UNIFIED DEEPFAKE DETECTOR (GAN + Diffusion)")
            print("="*70)
            print(f"\nImage: {image_path}")
            print(f"Output: {output_dir}")
            print(f"Max dimension: {self.max_dimension}px (auto-downsample)")
            print("\n" + "="*70)
        
        total_start = time.time()
        
        # Run GAN detector
        if self.verbose:
            print("\n[1/2] Running GAN Detector...")
            print("-"*70)
        
        gan_detector = OptimizedGANDetector(
            max_dimension=self.max_dimension,
            verbose=self.verbose
        )
        gan_results = gan_detector.analyze_image(
            image_path, 
            os.path.join(output_dir, "gan_analysis")
        )
        
        if "error" in gan_results:
            return {"error": f"GAN detector failed: {gan_results['error']}"}
        
        # Run Diffusion detector
        if self.verbose:
            print("\n[2/2] Running Diffusion Detector...")
            print("-"*70)
        
        diff_detector = OptimizedDiffusionDetector(
            max_dimension=self.max_dimension,
            verbose=self.verbose
        )
        diff_results = diff_detector.analyze_image(
            image_path,
            os.path.join(output_dir, "diffusion_analysis")
        )
        
        if "error" in diff_results:
            return {"error": f"Diffusion detector failed: {diff_results['error']}"}
        
        total_time = time.time() - total_start
        
        # Combine verdicts
        combined = self._combine_verdicts(gan_results, diff_results, total_time)
        
        if self.verbose:
            self._print_summary(combined)
        
        return combined
    
    def _combine_verdicts(self, gan_results, diff_results, total_time):
        """
        Intelligently combine both detector results.
        """
        gan_verdict = gan_results["verdict"]
        diff_verdict = diff_results["verdict"]
        
        gan_conf = gan_verdict["confidence"]
        diff_conf = diff_verdict["confidence"]
        
        # Decision logic
        if gan_conf >= 0.6 and diff_conf < 0.4:
            # Strong GAN signal, weak diffusion signal → Likely GAN
            final_verdict = "LIKELY_GAN_GENERATED"
            confidence = gan_conf
            explanation = f"Strong GAN artifacts detected (confidence: {gan_conf:.1%})"
            primary_detector = "GAN"
            
        elif diff_conf >= 0.6 and gan_conf < 0.4:
            # Strong diffusion signal, weak GAN signal → Likely Diffusion
            final_verdict = "LIKELY_DIFFUSION_MODEL"
            confidence = diff_conf
            explanation = f"Strong diffusion artifacts detected (confidence: {diff_conf:.1%})"
            primary_detector = "Diffusion"
            
        elif gan_conf >= 0.6 and diff_conf >= 0.6:
            # Both detectors flag it → Very suspicious, use higher confidence
            if gan_conf > diff_conf:
                final_verdict = "LIKELY_GAN_GENERATED"
                confidence = gan_conf
                explanation = f"Both detectors flagged, GAN artifacts stronger ({gan_conf:.1%} vs {diff_conf:.1%})"
                primary_detector = "GAN"
            else:
                final_verdict = "LIKELY_DIFFUSION_MODEL"
                confidence = diff_conf
                explanation = f"Both detectors flagged, diffusion artifacts stronger ({diff_conf:.1%} vs {gan_conf:.1%})"
                primary_detector = "Diffusion"
                
        elif gan_conf >= 0.4 or diff_conf >= 0.4:
            # At least one detector is suspicious
            final_verdict = "SUSPICIOUS"
            confidence = max(gan_conf, diff_conf)
            explanation = f"Suspicious signals detected (GAN: {gan_conf:.1%}, Diffusion: {diff_conf:.1%})"
            primary_detector = "GAN" if gan_conf > diff_conf else "Diffusion"
            
        else:
            # Both detectors show low confidence → Likely real
            final_verdict = "LIKELY_REAL"
            confidence = 1.0 - max(gan_conf, diff_conf)
            explanation = f"No strong artifacts detected (GAN: {gan_conf:.1%}, Diffusion: {diff_conf:.1%})"
            primary_detector = "None"
        
        return {
            "final_verdict": final_verdict,
            "confidence": confidence,
            "explanation": explanation,
            "primary_detector": primary_detector,
            "gan_results": {
                "verdict": gan_verdict["verdict"],
                "confidence": gan_conf,
                "signals": f"{gan_verdict['suspicious_signals']}/{gan_verdict['total_checks']}",
                "time": gan_results["total_processing_time"]
            },
            "diffusion_results": {
                "verdict": diff_verdict["verdict"],
                "confidence": diff_conf,
                "signals": f"{diff_verdict['suspicious_signals']}/{diff_verdict['total_checks']}",
                "time": diff_results["total_processing_time"]
            },
            "total_time": total_time,
            "image_info": {
                "original_size": gan_results["original_shape"],
                "processed_size": gan_results["processed_shape"],
                "resize_factor": gan_results["resize_factor"]
            }
        }
    
    def _print_summary(self, combined):
        """Print nice summary of results."""
        print("\n" + "="*70)
        print("UNIFIED ANALYSIS COMPLETE")
        print("="*70)
        
        print("\n📊 Individual Detector Results:")
        print("-"*70)
        
        print("\n  GAN Detector:")
        print(f"    Verdict: {combined['gan_results']['verdict']}")
        print(f"    Confidence: {combined['gan_results']['confidence']:.1%}")
        print(f"    Signals: {combined['gan_results']['signals']}")
        print(f"    Time: {combined['gan_results']['time']:.2f}s")
        
        print("\n  Diffusion Detector:")
        print(f"    Verdict: {combined['diffusion_results']['verdict']}")
        print(f"    Confidence: {combined['diffusion_results']['confidence']:.1%}")
        print(f"    Signals: {combined['diffusion_results']['signals']}")
        print(f"    Time: {combined['diffusion_results']['time']:.2f}s")
        
        print("\n" + "="*70)
        print("🎯 FINAL VERDICT")
        print("="*70)
        
        # Color code the verdict
        verdict = combined['final_verdict']
        if "LIKELY_GAN" in verdict or "LIKELY_DIFFUSION" in verdict:
            symbol = "🚨"
        elif "SUSPICIOUS" in verdict:
            symbol = "⚠️ "
        else:
            symbol = "✅"
        
        print(f"\n  {symbol} {verdict}")
        print(f"  Confidence: {combined['confidence']:.1%}")
        print(f"  Primary Signal: {combined['primary_detector']}")
        print(f"\n  {combined['explanation']}")
        
        print("\n" + "-"*70)
        print(f"  Total Processing Time: {combined['total_time']:.2f}s")
        print(f"  Image Size: {combined['image_info']['original_size']} → {combined['image_info']['processed_size']}")
        print(f"  Resize Factor: {combined['image_info']['resize_factor']:.2f}x")
        
        print("\n" + "="*70)
        print("\n💡 Interpretation Guide:")
        print("-"*70)
        if "GAN" in verdict:
            print("  • Image likely generated by GAN (StyleGAN, ProGAN, face-swap)")
            print("  • Check GAN analysis for specific artifacts (checkerboard, power law)")
        elif "DIFFUSION" in verdict:
            print("  • Image likely from diffusion model (Stable Diffusion, DALL-E, Midjourney)")
            print("  • Check diffusion analysis for noise residuals, texture repetition")
        elif "SUSPICIOUS" in verdict:
            print("  • Some artifacts detected, but not conclusive")
            print("  • Recommend manual review and additional verification")
        else:
            print("  • Few artifacts detected in frequency domain")
            print("  • Could be real photo OR very sophisticated fake")
            print("  • Consider: metadata analysis, reverse image search, expert review")
        
        print("\n⚠️  DISCLAIMER:")
        print("  This is a heuristic frequency-domain analysis, NOT definitive proof.")
        print("  Modern AI models (2024+) can fool these detectors.")
        print("  For critical decisions, use ML-based detectors + expert review.")
        print("="*70 + "\n")


def main():
    """Command line interface."""
    if len(sys.argv) < 2:
        print("""
╔════════════════════════════════════════════════════════════════════╗
║            UNIFIED FAST DEEPFAKE DETECTOR                          ║
║            (GAN + Diffusion Model Detection)                       ║
╚════════════════════════════════════════════════════════════════════╝

Usage:
  python unified_fast_detector.py <image_path> [output_dir] [max_size]

Examples:
  python unified_fast_detector.py suspicious_image.jpg
  python unified_fast_detector.py photo.png ./results
  python unified_fast_detector.py large_image.jpg ./results 1024

Options:
  image_path  - Path to image file (required)
  output_dir  - Output directory for results (default: ./unified_analysis)
  max_size    - Max image dimension in pixels (default: 1024)
                Images larger than this are automatically downsampled

Performance:
  • Large images (1622x2285): Auto-downsampled to ~1024x1024
  • Expected time: 5-15 seconds (vs 30+ min unoptimized)
  • Progress bars show real-time status

What it does:
  1. Runs GAN detector (checkerboard, power law, phase analysis)
  2. Runs Diffusion detector (noise residuals, texture repetition)
  3. Combines results intelligently
  4. Provides detailed visualizations

Output:
  • /gan_analysis/gan_analysis.png - GAN detector visualization
  • /diffusion_analysis/diffusion_analysis.png - Diffusion visualization
  • Console summary with final verdict
        """)
        sys.exit(1)
    
    image_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "./unified_analysis"
    max_dimension = int(sys.argv[3]) if len(sys.argv) > 3 else 1024
    
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"❌ Error: Image not found: {image_path}")
        sys.exit(1)
    
    # Run unified detector
    detector = UnifiedDeepfakeDetector(max_dimension=max_dimension, verbose=True)
    results = detector.analyze_image(image_path, output_dir)
    
    if "error" in results:
        print(f"\n❌ Error: {results['error']}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
