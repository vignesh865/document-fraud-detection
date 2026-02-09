# Strategy 4: Region-Based Frequency Domain Detection

## Overview

Combine Gemini semantic segmentation with frequency-domain detectors (GAN, Diffusion) applied individually to each region to identify AI-generated or manipulated content.

## Approach

**Three-Stage Pipeline**:
1. **Segmentation**: Gemini identifies document regions
2. **Per-Region Detection**: Run frequency analyzers on each region
3. **Aggregation**: Combine region-level verdicts

## Stage 1: Gemini Segmentation

Same as Strategy 3 - use Gemini to segment document into 30-44 semantic regions with type classification and descriptions.

## Stage 2: Frequency Analysis Per Region

### GAN Detector
**Purpose**: Detect GAN-generated or manipulated regions

**Method** (per region):
- Extract region image
- Compute 2D FFT (frequency spectrum)
- Analyze frequency patterns:
  - Checkerboard artifacts (upsampling)
  - Power law deviation (1/f distribution)
  - Azimuthal anisotropy
  - High-frequency anomalies
  - Phase coherence issues
- Score: 0-5 suspicious signals

**Verdict Levels**:
- 0-1 signals: LIKELY_REAL
- 2 signals: SUSPICIOUS  
- 3+ signals: LIKELY_FAKE

### Diffusion Detector
**Purpose**: Detect diffusion-model-generated regions

**Method** (per region):
- Local frequency inconsistency detection
- Texture repetition analysis
- Noise residual examination
- Cross-channel correlation (RGB)
- Multi-scale spectral analysis
- Score: 0-5 suspicious signals

**Verdict Levels**:
- 0-1 signals: LIKELY_REAL
- 2 signals: SUSPICIOUS
- 3+ signals: LIKELY_DIFFUSION_MODEL

## Stage 3: Aggregation

```
for region in regions:
    # Skip tiny regions
    if region.size < min_size:
        continue
    
    # Run detectors
    gan_verdict = gan_detector.analyze(region)
    diffusion_verdict = diffusion_detector.analyze(region)
    
    # Flag if either detector suspicious
    if gan_verdict in [SUSPICIOUS, LIKELY_FAKE] or 
       diffusion_verdict in [SUSPICIOUS, LIKELY_DIFFUSION_MODEL]:
        suspicious_regions.add(region)

# Document-level decision
suspicion_ratio = len(suspicious_regions) / len(analyzed_regions)

if suspicion_ratio > threshold:  # e.g., >30% regions flagged
    verdict = LIKELY_FAKE
```

## Key Advantages

- **Granular Detection**: Identifies which specific regions are AI/manipulated
- **Complementary Signals**: GAN and Diffusion catch different artifacts
- **Region Context**: Knows what type of content is being analyzed
- **Interpretable**: Can show user exactly what's suspicious
- **No Training Required**: Pre-built frequency analyzers

## Limitations

- **Content Type Sensitivity**: Logos/text trigger false positives
- **Compression Artifacts**: Original JPEG compression can trigger detection
- **Small Regions**: Frequency analysis needs reasonable image size
- **High Compute**: Running detectors on 30+ regions is slow
- **API Dependency**: Still needs Gemini for segmentation

## Critical Issue: Type-Aware Filtering

**Problem**: Running photo-trained detectors on text/logos causes false positives

**Solution**: Filter regions by type before analysis

```python
# Only analyze appropriate region types
analyzable_types = ['PHOTO', 'SEAL', 'WATERMARK']

for region in regions:
    if region.type in analyzable_types:
        # Run frequency detectors
    else:
        # Skip or use different analyzer
```

## Recommended Region Type Handling

| Region Type | Frequency Detection | Alternative Analysis |
|-------------|-------------------|---------------------|
| PHOTO | ✅ Yes (GAN/Diffusion) | Face analysis, lighting |
| SEAL | ✅ Yes (with caution) | Template matching |
| WATERMARK | ✅ Yes (with caution) | Presence validation |
| TEXT | ❌ No (use OCR) | Font analysis, rendering |
| MRZ | ❌ No (use OCR) | Checksum validation |
| LOGO | ❌ No (templates) | Template matching |
| BARCODE | ❌ No (decoder) | Format validation |
| BACKGROUND | ⚠️ Maybe | Texture consistency |

## Optimization Strategies

### Parallel Processing
- Run GAN and Diffusion detectors in parallel
- Process regions in parallel (thread pool)
- Cache detector initialization

### Early Termination
```python
if len(suspicious_regions) / len(regions) > 0.5:
    # Already exceeded threshold, stop analyzing
    return LIKELY_FAKE
```

### Progressive Analysis
1. Analyze high-priority regions first (PHOTO)
2. If clear verdict, skip remaining regions
3. If uncertain, analyze all regions

## Use Cases

### Ideal For
- Documents with AI-generated photos
- Detecting GAN-created faces in passports
- Mixed real/AI content detection
- Cases where specific region is suspect

### Less Effective For
- Fully scanned physical documents
- Documents with heavy compression
- Small/low-quality images
- Simple text-only documents

## Integration Points

Can be combined with:
- **Gemini semantic checks**: Frequency + logical validation
- **Embedding methods**: Use frequency as initial filter, embeddings for confirmation
- **Traditional CV**: ELA on suspicious frequency regions
- **Human review**: Frequency provides technical evidence

## Implementation Considerations

### Detector Selection
- **GAN detector**: Better for upsampling artifacts
- **Diffusion detector**: Better for texture patterns
- **Both**: Maximum coverage but slower

### Region Preprocessing
- Resize very large regions to standard size
- Apply minimal filtering to preserve artifacts
- Extract at original quality (no re-compression)

### Threshold Tuning
Challenge: Set suspicion_ratio threshold
- Conservative (30-40%): Fewer false positives
- Aggressive (10-20%): Higher sensitivity
- Adaptive: Based on region types present

### Caching
- Cache frequency analysis per region
- Use region content hash for cache key
- Significant speedup on re-analysis
