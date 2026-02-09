# Frequency-Domain Deepfake Detector

A comprehensive frequency-domain analysis tool for detecting AI-generated images and deepfakes.

## Overview

This detector uses multiple frequency-domain techniques to identify artifacts commonly found in GAN-generated images. While modern GANs have become very sophisticated, frequency analysis remains a valuable tool in the deepfake detection arsenal.

## Key Features

### 1. **Checkerboard Artifact Detection**
- **What it detects**: Periodic patterns caused by transposed convolution (deconvolution) layers in GANs
- **How**: Analyzes radial frequency profile for periodic peaks
- **Why it matters**: Early and mid-generation GANs (2014-2019) commonly produce these artifacts

### 2. **Power Law Analysis (1/f Distribution)**
- **What it detects**: Deviations from natural image statistics
- **How**: Fits power spectral density to 1/f^α and checks if α ≈ 2.0
- **Why it matters**: Natural images follow specific power law distributions; deviations suggest manipulation

### 3. **Azimuthal (Angular) Analysis**
- **What it detects**: Directional artifacts in frequency domain
- **How**: Analyzes variance in angular frequency distribution
- **Why it matters**: Natural images are roughly isotropic; GAN artifacts can create anisotropic patterns

### 4. **High-Frequency Artifact Detection**
- **What it detects**: Abnormal high-frequency content
- **How**: Compares energy ratios across frequency bands
- **Why it matters**: Some GANs produce too-smooth images (low HF) or compression artifacts (high HF)

### 5. **Phase Coherence Analysis**
- **What it detects**: Non-natural phase distributions
- **How**: Analyzes entropy and variance of phase spectrum
- **Why it matters**: Many GAN artifacts appear in phase, not just magnitude

## Installation

```bash
# Required packages
pip install opencv-python numpy matplotlib scipy
```

## Usage

### Basic Usage

```bash
python frequency_deepfake_detector.py image.jpg output_dir
```

### Python API

```python
from frequency_deepfake_detector import FrequencyDeepfakeDetector

detector = FrequencyDeepfakeDetector()
results = detector.analyze_image("image.jpg", "output_dir")

print(results["verdict"])
# {
#     "verdict": "LIKELY_FAKE" | "SUSPICIOUS" | "LIKELY_REAL",
#     "confidence": 0.8,
#     "suspicious_signals": 4,
#     "total_checks": 5,
#     ...
# }
```

### Running Tests

```bash
python test_detector.py
```

This will:
1. Create synthetic test images with known artifacts
2. Run analysis on each image
3. Generate comprehensive visualizations
4. Save results to `test_results/` directory

## Understanding the Results

### Verdict Categories

1. **LIKELY_FAKE** (confidence ≥ 60%)
   - Multiple suspicious signals detected
   - High probability of manipulation
   - Recommend rejecting or further investigation

2. **SUSPICIOUS** (40% ≤ confidence < 60%)
   - Some anomalies detected
   - Further analysis recommended
   - Could be JPEG artifacts, low quality, or actual manipulation

3. **LIKELY_REAL** (confidence < 40%)
   - Few or no anomalies detected
   - Appears natural in frequency domain
   - Note: Modern GANs can still pass

### Individual Test Results

#### 1. Checkerboard Score
- **Range**: 0.0 - 1.0
- **Threshold**: > 0.3 is suspicious
- **Interpretation**:
  - **High score**: Periodic peaks in frequency domain → upsampling artifacts
  - **Low score**: Smooth radial profile → natural or advanced GAN

#### 2. Power Law Alpha (α)
- **Expected**: 1.5 - 2.5 (typically ~2.0)
- **Threshold**: Deviation > 0.7 is suspicious
- **Interpretation**:
  - **α too low** (< 1.3): Not enough high-frequency fall-off → unnatural
  - **α too high** (> 2.7): Too much fall-off → over-smoothed
  - **Good R²** (> 0.85): Confirms reliable fit

#### 3. Anisotropy Score
- **Range**: 0.0 - 1.0+
- **Threshold**: > 0.25 is suspicious
- **Interpretation**:
  - **High score**: Directional bias in frequencies → artifacts
  - **Low score**: Isotropic distribution → natural

#### 4. High/Low Frequency Ratio
- **Expected**: 0.01 - 0.15
- **Interpretation**:
  - **Too high** (> 0.15): Excessive high-frequency → JPEG, noise, artifacts
  - **Too low** (< 0.01): Too smooth → over-processed GAN
  - **In range**: Natural distribution

#### 5. Phase Entropy
- **Expected**: > 0.85 (normalized)
- **Threshold**: < 0.85 is suspicious
- **Interpretation**:
  - **Low entropy**: Non-uniform phase distribution → manipulation
  - **High entropy**: Natural phase distribution

## Limitations and Disclaimers

### ⚠️ Important Limitations

1. **Modern GANs (2020+)**: Many sophisticated GANs can fool these tests
   - StyleGAN2, StyleGAN3, DALL-E, Midjourney, Stable Diffusion
   - These models have addressed many frequency-domain artifacts

2. **Image Quality Matters**:
   - JPEG compression creates high-frequency artifacts (false positives)
   - Low resolution reduces detection accuracy
   - Post-processing can mask or introduce artifacts

3. **Not Production-Ready**:
   - This is a **heuristic** approach, not ML-based
   - Production systems should use:
     - Deep learning classifiers
     - Ensemble methods
     - Multi-modal analysis (frequency + spatial + metadata)

4. **False Positives**:
   - Heavily compressed images
   - Low-quality cameras
   - Certain artistic styles
   - Heavily edited real photos

5. **False Negatives**:
   - State-of-the-art GANs
   - Post-processed synthetic images
   - Hybrid real/fake images

## When to Use This Tool

### ✅ Good Use Cases

- **Educational purposes**: Understanding frequency-domain artifacts
- **Preliminary screening**: First-pass filter for obvious fakes
- **Forensic analysis**: One signal among many in investigation
- **Research**: Studying GAN artifact evolution
- **Legacy deepfakes**: Detecting older GAN outputs (2014-2019)

### ❌ Not Recommended For

- **Critical decisions**: Don't rely solely on this for important determinations
- **Legal evidence**: Needs expert testimony and validation
- **Modern AI art**: Will likely produce false positives
- **Production systems**: Use ML-based detection instead

## Technical Details

### Frequency Domain Analysis

The Discrete Fourier Transform (DFT) converts an image from spatial domain to frequency domain:

```
F(u,v) = Σ Σ f(x,y) * e^(-2πi(ux/M + vy/N))
```

Where:
- `f(x,y)` is the spatial domain image
- `F(u,v)` is the frequency domain representation
- `u,v` are frequency coordinates
- `M,N` are image dimensions

### Power Law in Natural Images

Natural images exhibit 1/f^α power spectral density:

```
PSD(f) ∝ 1/f^α
```

Where:
- `f` is spatial frequency
- `α ≈ 2` for most natural images (pink noise)
- Deviations indicate non-natural content

### Checkerboard Artifacts

Caused by transposed convolution (deconvolution) used in GAN upsampling:

```
output[i,j] = Σ Σ input[i/s, j/s] * kernel[i%s, j%s]
```

This creates periodic patterns visible in frequency domain.

## Research References

This implementation is based on research including:

1. **Durall et al. (2020)**: "Watch your Up-Convolution: CNN Based Generative Deep Neural Networks are Failing to Reproduce Spectral Distributions"

2. **Frank et al. (2020)**: "Leveraging Frequency Analysis for Deep Fake Image Recognition"

3. **Dzanic et al. (2020)**: "Fourier Spectrum Discrepancies in Deep Network Generated Images"

4. **Zhang et al. (2019)**: "Detecting and Simulating Artifacts in GAN Fake Images"

## Example Output

When you run the detector, you get:

1. **Console output**: Detailed text report of all analyses
2. **Visualization**: Comprehensive multi-panel figure showing:
   - Original image
   - Magnitude spectrum
   - Phase spectrum
   - Radial profile (checkerboard detection)
   - Power law fit
   - Angular profile
   - Frequency band energies
   - Phase entropy
   - Overall verdict with confidence
   - Detailed check results

## Contributing

Potential improvements:

- [ ] Add more sophisticated peak detection algorithms
- [ ] Implement sliding window analysis for localized artifacts
- [ ] Add comparison with database of known GAN signatures
- [ ] Integrate ML classifier trained on frequency features
- [ ] Add color channel analysis (RGB vs. frequency correlation)
- [ ] Implement real-time video analysis
- [ ] Add metadata extraction and analysis

## License

This is educational/research code. Use responsibly.

## Acknowledgments

Built upon research from the computer vision and deepfake detection community.

---

**Remember**: This tool is one piece of evidence, not definitive proof. Always use multiple detection methods and human expertise for important decisions.
