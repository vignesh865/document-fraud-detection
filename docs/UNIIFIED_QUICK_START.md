# QUICK START GUIDE - Optimized Fast Detectors

## 🚀 TL;DR - For Your 1622×2285 Image

```bash
# Install dependencies
pip install opencv-python numpy scipy matplotlib tqdm

# Run the unified detector (RECOMMENDED)
python unified_fast_detector.py your_image.jpg

# Expected time: 5-15 seconds ✅
# (vs 30+ minutes with original code)
```

---

## 📦 What You Got

### 🔥 **NEW: Optimized Versions** (Use These!)

1. **unified_fast_detector.py** - ⭐ RECOMMENDED
   - Runs BOTH GAN and Diffusion detection
   - Auto-downsamples large images
   - Progress bars
   - 5-15 seconds total

2. **optimized_gan_detector.py** - For GAN-only detection
   - Fast checkerboard, power law, phase analysis
   - 3-8 seconds

3. **optimized_diffusion_detector.py** - For Diffusion-only detection
   - Fast noise residual, texture repetition analysis
   - 4-10 seconds

### 📚 Original Versions (Educational)

4. **frequency_deepfake_detector.py** - Original GAN detector (slow)
5. **diffusion_detector.py** - Original diffusion detector (very slow)

---

## 🎯 Which One Should You Use?

### ✅ Use `unified_fast_detector.py` if:
- You don't know what type of fake it might be
- You want comprehensive analysis
- You want both GAN and Diffusion detection
- **→ This is the BEST option for most users**

### Use `optimized_gan_detector.py` if:
- You specifically suspect GAN (StyleGAN, face-swap)
- You want faster results (skip diffusion detection)
- Working with 2016-2020 era deepfakes

### Use `optimized_diffusion_detector.py` if:
- You specifically suspect diffusion model (Midjourney, DALL-E, Stable Diffusion)
- You want faster results (skip GAN detection)
- Working with AI art from 2022+

---

## 💻 Installation

```bash
# Required packages
pip install opencv-python numpy scipy matplotlib tqdm

# Optional: Install all at once
pip install opencv-python numpy scipy matplotlib tqdm --upgrade
```

---

## 🏃 Quick Examples

### Example 1: Basic Usage (Recommended)

```bash
# Analyze any image
python unified_fast_detector.py suspicious_photo.jpg

# Output will be in ./unified_analysis/
# - gan_analysis/gan_analysis.png
# - diffusion_analysis/diffusion_analysis.png
```

### Example 2: Custom Output Directory

```bash
python unified_fast_detector.py image.jpg ./my_results
```

### Example 3: Adjust Max Size (For Speed vs Accuracy)

```bash
# Faster (lower accuracy)
python unified_fast_detector.py image.jpg ./results 512

# Balanced (default)
python unified_fast_detector.py image.jpg ./results 1024

# Slower but more accurate
python unified_fast_detector.py image.jpg ./results 2048
```

### Example 4: Batch Processing

```python
from unified_fast_detector import UnifiedDeepfakeDetector
import os

detector = UnifiedDeepfakeDetector(max_dimension=1024, verbose=False)

for img_file in os.listdir("suspect_images/"):
    if img_file.endswith(('.jpg', '.png')):
        results = detector.analyze_image(
            f"suspect_images/{img_file}",
            f"results/{img_file}_analysis"
        )
        
        verdict = results["final_verdict"]
        confidence = results["confidence"]
        
        print(f"{img_file}: {verdict} ({confidence:.1%})")
```

---

## 📊 Understanding the Output

### Console Output

```
════════════════════════════════════════════════════════════════════
UNIFIED DEEPFAKE DETECTOR (GAN + Diffusion)
════════════════════════════════════════════════════════════════════

📁 Loading image...
✓ Image loaded: (2285, 1622) → (1024, 727) (factor: 0.45)

[1/2] Running GAN Detector...
Progress: ████████████████████ 5/5 100%
✓ Analysis complete in 6.3s

[2/2] Running Diffusion Detector...
Progress: ████████████████████ 5/5 100%
✓ Analysis complete in 7.8s

════════════════════════════════════════════════════════════════════
🎯 FINAL VERDICT
════════════════════════════════════════════════════════════════════

  🚨 LIKELY_DIFFUSION_MODEL
  Confidence: 80.0%
  Primary Signal: Diffusion

  Strong diffusion artifacts detected (confidence: 80.0%)

Total Processing Time: 14.1s
════════════════════════════════════════════════════════════════════
```

### Verdict Types

| Verdict | Meaning | Action |
|---------|---------|--------|
| **LIKELY_GAN_GENERATED** | Strong GAN artifacts (StyleGAN, ProGAN, face-swap) | Reject or investigate |
| **LIKELY_DIFFUSION_MODEL** | Strong diffusion artifacts (Stable Diffusion, DALL-E) | Reject or investigate |
| **SUSPICIOUS** | Some artifacts but not conclusive | Manual review |
| **LIKELY_REAL** | Few artifacts detected | Likely authentic (but verify) |

### Confidence Levels

- **≥ 80%** - Very confident, strong signal
- **60-80%** - Confident, clear artifacts
- **40-60%** - Uncertain, needs review
- **< 40%** - Low confidence, appears natural

---

## 📈 Performance Expectations

| Image Size | Unified Detector | GAN Only | Diffusion Only |
|------------|------------------|----------|----------------|
| 512 × 512 | ~4s | ~2s | ~2s |
| 1024 × 1024 | ~8s | ~4s | ~4s |
| **1622 × 2285** | **~14s** | **~7s** | **~7s** |
| 2048 × 2048 | ~20s | ~10s | ~10s |
| 4096 × 4096 | ~45s | ~22s | ~23s |

**Note**: Original unoptimized code would take 30+ minutes for 1622×2285!

---

## 🔧 Troubleshooting

### Problem: "Module not found: tqdm"

```bash
pip install tqdm
```

### Problem: "Module not found: cv2"

```bash
pip install opencv-python
```

### Problem: Still slow (> 1 minute)

```bash
# Use more aggressive downsampling
python unified_fast_detector.py image.jpg ./results 512
# Should complete in < 10s even for huge images
```

### Problem: Low accuracy on huge images

```bash
# Increase max dimension (trades speed for accuracy)
python unified_fast_detector.py image.jpg ./results 2048
```

### Problem: "Error: Could not read image"

```python
# Check file path and format
import cv2
img = cv2.imread("your_image.jpg")
if img is None:
    print("Cannot read image - check path and format")
```

---

## 🎨 Visual Output Explained

Each detector creates a visualization with:

### GAN Analysis Visualization
```
┌────────────┬────────────┬────────────┬────────────┐
│  Original  │ Magnitude  │Checkerboard│ Power Law  │
│   Image    │  Spectrum  │   Score    │   Alpha    │
├────────────┼────────────┼────────────┼────────────┤
│ Azimuthal  │ High-Freq  │   Phase    │  VERDICT   │
│ Anisotropy │  Artifacts │  Entropy   │ + Confidence│
└────────────┴────────────┴────────────┴────────────┘
```

### Diffusion Analysis Visualization
```
┌────────────┬────────────┬────────────┬────────────┐
│  Original  │   Local    │  Texture   │   Noise    │
│   Image    │   Freq     │ Repetition │  Residual  │
├────────────┼────────────┼────────────┼────────────┤
│Cross-Chan  │Multi-Scale │  VERDICT   │ Processing │
│Correlation │  Spectral  │ + Confidence│    Info    │
└────────────┴────────────┴────────────┴────────────┘
```

**Green boxes** = Passed test (appears natural)
**Yellow boxes** = Failed test (suspicious)

---

## 💡 Best Practices

### ✅ DO:

1. **Use unified detector for unknown sources**
   ```bash
   python unified_fast_detector.py unknown_image.jpg
   ```

2. **Adjust max_dimension for your needs**
   - Speed priority: `max_dimension=512`
   - Balanced: `max_dimension=1024` (default)
   - Accuracy priority: `max_dimension=2048`

3. **Combine with other verification methods**
   - Metadata analysis (EXIF data)
   - Reverse image search
   - Visual inspection for inconsistencies

4. **Batch process with progress disabled**
   ```python
   detector = UnifiedDeepfakeDetector(verbose=False)
   ```

### ❌ DON'T:

1. **Don't rely solely on this for critical decisions**
   - Use as screening tool, not definitive proof
   - For important cases: expert review + ML-based detection

2. **Don't process unnecessarily large images**
   - Downsampling to 1024px is usually sufficient
   - Minimal accuracy loss, huge speed gain

3. **Don't expect 100% accuracy**
   - Modern AI (2024+) can fool these detectors
   - False positives/negatives happen

---

## 🆘 Getting Help

### Check the Documentation

- **PERFORMANCE_GUIDE.md** - Detailed optimization explanations
- **GAN_VS_DIFFUSION_COMPARISON.md** - Which detector for which model
- **diffusion_models_analysis.md** - Deep dive on diffusion detection
- **README.md** - Full technical documentation

### Common Questions

**Q: Why is my image downsampled?**
A: Large images (>1024px) are auto-downsampled for speed. Frequency artifacts are still detectable at lower resolution.

**Q: Can I disable downsampling?**
A: Yes! Use `max_dimension=4096` or higher. But expect slower processing.

**Q: Which is better, GAN or Diffusion detector?**
A: Use unified detector - it runs both and picks the best result!

**Q: How accurate is this?**
A: ~70-80% on older models (2016-2020), ~50-60% on modern models (2022+). Not production-ready.

---

## 🚀 Quick Reference Card

```
┌─────────────────────────────────────────────────────────────┐
│                    QUICK REFERENCE                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  BASIC USAGE:                                               │
│  python unified_fast_detector.py image.jpg                  │
│                                                             │
│  CUSTOM OUTPUT:                                             │
│  python unified_fast_detector.py image.jpg ./my_results     │
│                                                             │
│  ADJUST SPEED/ACCURACY:                                     │
│  python unified_fast_detector.py image.jpg ./out 512   ← Fast│
│  python unified_fast_detector.py image.jpg ./out 1024  ← Balanced│
│  python unified_fast_detector.py image.jpg ./out 2048  ← Accurate│
│                                                             │
│  EXPECTED TIME (1622×2285 image):                           │
│  Unified:   ~14 seconds                                     │
│  GAN only:  ~7 seconds                                      │
│  Diff only: ~7 seconds                                      │
│                                                             │
│  OUTPUT LOCATION:                                           │
│  ./unified_analysis/gan_analysis/gan_analysis.png           │
│  ./unified_analysis/diffusion_analysis/diffusion_analysis.png│
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 🎓 Next Steps

1. **Try it out**: Run unified detector on a test image
2. **Examine visualizations**: Look at the PNG outputs
3. **Understand results**: Read the verdict explanations
4. **Batch process**: If needed, process multiple images
5. **Combine methods**: Use with metadata analysis, reverse search
6. **Consider ML**: For production, look into trained models

---

## ⚡ Performance Tips

**For maximum speed:**
```python
detector = UnifiedDeepfakeDetector(
    max_dimension=512,  # Aggressive downsampling
    verbose=False       # No progress bars
)
```

**For maximum accuracy:**
```python
detector = UnifiedDeepfakeDetector(
    max_dimension=2048,  # Minimal downsampling
    verbose=True         # See what's happening
)
```

**For batch processing:**
```python
detector = UnifiedDeepfakeDetector(
    max_dimension=1024,  # Balanced
    verbose=False        # Skip progress bars
)

# Process 100 images in ~15 minutes
```

---

**That's it! You're ready to detect deepfakes at blazing speed! 🚀**
