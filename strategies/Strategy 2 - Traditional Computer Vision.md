# Strategy 2: Traditional Computer Vision Analysis

## Overview

Use classical image processing and frequency-domain analysis techniques to detect manipulation artifacts without AI/ML models or API dependencies.

## Approach

Apply multiple independent computer vision analyzers to the entire document, each targeting specific manipulation artifacts. Combine signals to reach final verdict.

## Core Analyzers

### 1. Error Level Analysis (ELA)
**Purpose**: Detect compression inconsistencies from editing

**Method**:
- Re-save image at known JPEG quality
- Compute difference between original and re-saved
- Regions with different compression levels show different error magnitudes
- Edited areas often have higher error levels

**Detects**: Copy-paste, photo swaps, overlaid content

### 2. Noise Variance Analysis
**Purpose**: Detect inconsistent noise patterns

**Method**:
- Extract noise by subtracting denoised version from original
- Analyze noise variance across image patches
- Calculate coefficient of variation
- Detect outlier patches

**Detects**: Spliced regions, different camera sources, smoothing artifacts

### 3. PNG-Specific Analysis
**Purpose**: Detect PNG manipulation via LSB, chunks, statistics

**Method**:
- **LSB Analysis**: Check least significant bits for hidden data
- **Chunk Analysis**: Examine PNG metadata chunks for editor signatures
- **Statistical Analysis**: Find pixel distribution anomalies from splicing

**Detects**: Steganography, editor metadata, statistical splice artifacts

### 4. Font Consistency Analysis
**Purpose**: Detect text overlay or replacement

**Method**:
- Extract text regions via edge detection
- Analyze font characteristics (stroke width, anti-aliasing)
- Measure consistency across text elements
- Detect kerning/spacing irregularities

**Detects**: Overlaid text, font substitution, manually edited fields

### 5. Resolution Inconsistency Detection
**Purpose**: Detect pasted content at different resolutions

**Method**:
- Analyze local resolution indicators (edge sharpness)
- Identify sharp boundaries between quality zones
- Detect resampling artifacts
- Find upscaled/downscaled regions

**Detects**: Content from different sources, resolution mismatches

### 6. Frequency Domain Analysis (PRNU)
**Purpose**: Detect camera/scanner inconsistencies

**Method**:
- Extract Photo Response Non-Uniformity (sensor noise pattern)
- Check if noise pattern is consistent across document
- Detect foreign content with different PRNU

**Detects**: Content from different cameras/scanners

## Decision Logic

Each analyzer returns:
- `is_suspicious`: Boolean flag
- Quantitative metrics
- Suspicious regions/areas

**Aggregation**:
```
suspicious_count = sum(analyzer.is_suspicious for analyzer in analyzers)

if suspicious_count >= threshold_high:
    verdict = LIKELY_FAKE
elif suspicious_count >= threshold_medium:
    verdict = SUSPICIOUS
else:
    verdict = LIKELY_REAL
```



## Limitations

- **Threshold Sensitivity**: Hard to set universal thresholds
- **False Positives**: Natural variations trigger detection
- **Limited Semantic Understanding**: Can't validate logic/checksums
- **Document-Agnostic**: Doesn't understand document structure
- **Compressed Images**: Artifacts from original compression cause noise

## Use Cases



## Threshold Tuning

Critical challenge: Setting thresholds that balance false positives/negatives

**Strategies**:
- Conservative (low FP): High thresholds, may miss fakes
- Aggressive (low FN): Low thresholds, many false positives
- Adaptive: Different thresholds per document type
- Ensemble: Weight multiple weak signals

## Integration Points

Can be combined with:
- Gemini AI (CV for artifacts, AI for semantics)
- Region-based analysis (apply CV per region)
- Embedding methods (CV for screening, embeddings for confirmation)
- Human review (CV provides technical evidence)

## Implementation Notes

- Each analyzer independent (parallelizable)
- Visualizations help human interpretation
- Metrics useful for training ML classifiers later
- Can build document-type-specific profiles
