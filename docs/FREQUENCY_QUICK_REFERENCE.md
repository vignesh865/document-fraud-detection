# Quick Reference Guide

## Installation & Setup

```bash
# 1. Install dependencies
pip install opencv-python numpy matplotlib scipy

# 2. Download the detector
# (detector files should be in same directory)
```

## Quick Start

### Analyze a Single Image

```bash
python frequency_deepfake_detector.py my_image.jpg ./results
```

### Run Test Suite

```bash
python test_detector.py
```

---

## Reading the Results

### Understanding the Verdict

| Verdict | Confidence | Meaning | Action |
|---------|-----------|---------|--------|
| **LIKELY_FAKE** | ≥60% | Multiple red flags | Reject or investigate further |
| **SUSPICIOUS** | 40-60% | Some anomalies | Manual review recommended |
| **LIKELY_REAL** | <40% | Few anomalies | Likely authentic (but verify) |

### Key Metrics Cheat Sheet

#### ✅ What "Good" (Natural) Looks Like

```
Checkerboard Score:    < 0.3
Power Law Alpha (α):   1.5 - 2.5
Anisotropy:           < 0.25
High/Low Ratio:       0.01 - 0.15
Phase Entropy:        > 0.85
```

#### ⚠️ What "Suspicious" Looks Like

```
Checkerboard Score:    > 0.3    (upsampling artifacts)
Power Law Alpha:      < 1.3     (too little high-freq)
                      > 2.7     (too much high-freq)
Anisotropy:           > 0.25    (directional bias)
High/Low Ratio:       > 0.15    (too noisy)
                      < 0.01    (too smooth)
Phase Entropy:        < 0.85    (non-natural phase)
```

---

## Common Scenarios

### Scenario 1: Social Media Profile Photo

```bash
python frequency_deepfake_detector.py profile.jpg ./analysis
```

**Expect:**
- High checkerboard score if face-swapped
- Low high/low ratio if StyleGAN-generated
- Look for: α deviation, phase anomalies

### Scenario 2: News Article Image

```bash
python frequency_deepfake_detector.py news_image.jpg ./analysis
```

**Expect:**
- May have JPEG artifacts (false positives)
- Check power law first (most reliable)
- Look for: periodic patterns, phase issues

### Scenario 3: Batch Analysis

```python
from frequency_deepfake_detector import FrequencyDeepfakeDetector
import os

detector = FrequencyDeepfakeDetector()

for img_file in os.listdir("images/"):
    if img_file.endswith(('.jpg', '.png')):
        results = detector.analyze_image(
            os.path.join("images/", img_file),
            os.path.join("results/", img_file.replace('.', '_'))
        )
        
        if results["verdict"]["confidence"] >= 0.6:
            print(f"⚠️  {img_file}: {results['verdict']['verdict']}")
```

---

## Interpreting Visualizations

### Panel Guide

```
┌─────────────────────────────────────────────────────┐
│ [1] Original    [2] Magnitude   [3] Phase  [4] Radial│
│                      Spectrum    Spectrum   Profile  │
│                                                       │
│ [5] Power Law   [6] Angular     [7] Band   [8] Phase │
│     Analysis        Profile      Energy     Entropy  │
│                                                       │
│ [9] VERDICT    [10] Checks    [11] Explain [12] Notes│
└─────────────────────────────────────────────────────┘
```

#### What to Look For

**Panel 2 (Magnitude Spectrum)**:
- Natural: Bright center, smooth fall-off
- Fake: Grid patterns, periodic bright spots, sharp transitions

**Panel 4 (Radial Profile)**:
- Natural: Smooth exponential decay
- Fake: Bumps, peaks, oscillations

**Panel 6 (Angular Profile)**:
- Natural: Relatively flat/uniform
- Fake: Strong variations, directional bias

**Panel 7 (Band Energy)**:
- Natural: Green (low) >> Orange (mid) >> Red (high)
- Fake: Unusual ratios, inverted pattern

---

## Advanced Usage

### Custom Thresholds

```python
from frequency_deepfake_detector import FrequencyDeepfakeDetector

detector = FrequencyDeepfakeDetector()

# Analyze
results = detector.analyze_image("image.jpg")

# Custom decision logic
checkerboard = results["checkerboard_score"]["score"]
power_law = results["power_law_analysis"]["alpha_deviation"]

if checkerboard > 0.5 and power_law > 1.0:
    print("VERY LIKELY FAKE")
elif checkerboard > 0.3 or power_law > 0.7:
    print("SUSPICIOUS")
else:
    print("LIKELY REAL")
```

### Extract Specific Metrics

```python
results = detector.analyze_image("image.jpg")

# Get just what you need
alpha = results["power_law_analysis"]["alpha"]
confidence = results["verdict"]["confidence"]

print(f"Power law exponent: {alpha:.3f}")
print(f"Overall confidence: {confidence:.1%}")
```

---

## Troubleshooting

### Problem: Everything shows "LIKELY_FAKE"

**Possible Causes:**
- Images are heavily JPEG compressed
- Images are low resolution (< 256x256)
- Images are from low-quality camera

**Solutions:**
- Use higher quality source images
- Adjust thresholds for your use case
- Combine with other detection methods

### Problem: Known fakes show "LIKELY_REAL"

**Possible Causes:**
- Modern GAN (StyleGAN3, DALL-E 3, etc.)
- Post-processed to remove artifacts
- High-quality synthetic image

**Solutions:**
- This is expected! Modern GANs fool frequency analysis
- Use ML-based detectors for modern content
- Look at metadata, eye reflections, other signals

### Problem: "Could not read image" error

**Causes:**
- Invalid file path
- Corrupted image file
- Unsupported format

**Solutions:**
```python
import cv2
img = cv2.imread("path/to/image.jpg")
if img is None:
    print("Cannot read image!")
else:
    print(f"Image shape: {img.shape}")
```

---

## Best Practices

### ✅ DO:

1. **Use as one signal among many**
   - Combine with metadata analysis
   - Check for inconsistencies (lighting, shadows)
   - Look at physiological impossibilities

2. **Consider image quality**
   - Higher resolution = better accuracy
   - Uncompressed > JPEG
   - Original > downloaded/reposted

3. **Understand your use case**
   - Detecting 2015 GANs? This works well
   - Detecting 2024 GANs? Use ML methods

4. **Document your findings**
   - Save the visualization
   - Note which specific checks failed
   - Record confidence levels

### ❌ DON'T:

1. **Don't rely solely on this tool**
   - Not a magic bullet
   - False positives happen
   - False negatives happen

2. **Don't ignore context**
   - Check source credibility
   - Verify with reverse image search
   - Look for logical inconsistencies

3. **Don't use for critical decisions alone**
   - Legal matters need expert analysis
   - News verification needs multiple sources
   - Security applications need robust systems

---

## Performance Tips

### For Faster Processing

```python
# Skip visualization generation
results = detector.analyze_image("image.jpg", output_dir=None)
```

### For Batch Processing

```python
# Process multiple images efficiently
detector = FrequencyDeepfakeDetector()  # Reuse instance

for img_path in image_paths:
    results = detector.analyze_image(img_path)
    # Process results...
```

---

## When to Escalate

**Escalate to Expert Review if:**

1. High-stakes decision (legal, financial, security)
2. Mixed signals (some tests pass, some fail)
3. Confidence in middle range (40-60%)
4. Image is from critical source
5. Consequences of error are severe

**Consider ML-Based Detection if:**

1. Dealing with modern content (2020+)
2. Need high accuracy on recent GANs
3. Processing large volumes
4. Building production system
5. Face-swap or video deepfakes

---

## Resources

### Learn More About:

- **Deepfake Detection**: https://github.com/topics/deepfake-detection
- **Frequency Analysis**: OpenCV FFT documentation
- **GAN Artifacts**: Papers by Durall et al., Frank et al.

### Tools to Combine With:

- Metadata analysis (ExifTool)
- Reverse image search (TinEye, Google Images)
- Eye reflection analysis
- Physiological consistency checks
- ML-based detectors (if available)

---

## Quick Decision Tree

```
Is image from trusted source?
├─ Yes → Likely real, verify if critical
└─ No → Continue analysis
    │
    Run frequency detector
    │
    Confidence ≥ 60%?
    ├─ Yes → LIKELY FAKE
    │   └─ Action: Reject or investigate
    │
    Confidence 40-60%?
    ├─ Yes → SUSPICIOUS
    │   └─ Action: Manual review + other checks
    │
    Confidence < 40%?
    └─ Yes → LIKELY REAL
        └─ Action: Additional verification if critical
```

---

## Remember

> "All models are wrong, but some are useful."
> - George Box

This tool is useful for **understanding** frequency-domain artifacts and **screening** for obvious fakes, but it's not infallible. Always use multiple methods and human judgment.
