# Strategy 5: Embedding-Based Region Consistency Detection

## Overview

Use visual embedding models (DINOv2, CLIP) to detect inconsistencies between document regions that indicate composite forgeries. Based on the premise that authentic documents have consistent embeddings while pasted/edited regions show embedding discontinuities.

## Core Hypothesis

**Authentic documents**: All regions from same source → consistent embeddings  
**Forged documents**: Pasted regions from different sources → inconsistent embeddings

## Approach

**Three-Stage Pipeline**:
1. **Segmentation**: Gemini identifies semantic regions
2. **Embedding Extraction**: Compute visual embeddings for each region
3. **Consistency Analysis**: Detect regions with anomalous embeddings

## Stage 1: Gemini Segmentation

Use Gemini to segment document into regions with semantic labels (same as Strategies 3-4).

## Stage 2: Embedding Extraction

### Model Selection

**DINOv2** (Recommended for Region Consistency):
- Self-supervised visual features
- Excellent for capturing image characteristics
- Fast inference
- Good for within-document comparisons

**CLIP** (Alternative for Cross-Modal):
- Joint vision-language embeddings
- Can compare images to text descriptions
- Slower but more semantic

### Per-Region Embedding
```python
for region in regions:
    region_image = extract_region(document, region.bbox)
    embedding = model.encode(region_image)
    
    region_embeddings.append({
        'region_id': region.id,
        'type': region.type,
        'embedding': embedding  # Dense vector (768D for DINOv2)
    })
```

## Stage 3: Consistency Analysis

### Method 1: Global Consistency Score

Compare each region to all other regions:

```python
for region in embeddings:
    # Compute similarity to all other regions
    similarities = []
    for other in embeddings:
        if region != other:
            sim = cosine_similarity(region.embedding, other.embedding)
            similarities.append(sim)
    
    # Low average similarity = suspicious
    consistency_score = mean(similarities)
    
    if consistency_score < threshold:
        suspicious_regions.add(region)
```

**Interpretation**: Regions with low consistency likely from different source.

### Method 2: Spatial Locality Check

Compare regions only to spatial neighbors:

```python
for region in regions:
    neighbors = get_adjacent_regions(region, regions)
    
    neighbor_sims = [
        cosine_similarity(region.embedding, n.embedding)
        for n in neighbors
    ]
    
    # Neighbors should be MORE similar than distant regions
    local_consistency = mean(neighbor_sims)
    
    # Also compare to distant regions
    distant_regions = get_distant_regions(region, regions)
    distant_sims = [
        cosine_similarity(region.embedding, d.embedding)
        for d in distant_regions
    ]
    global_consistency = mean(distant_sims)
    
    # Neighbors should have higher similarity
    if local_consistency < global_consistency:
        # Anomaly: region doesn't fit with neighbors!
        suspicious_regions.add(region)
```

**Interpretation**: Pasted regions show discontinuity with immediate surroundings.

### Method 3: Multi-Scale Consistency

Check if embedding similarity preserved across different scales:

```python
for region in regions:
    # Get embeddings at different scales
    full_scale = model.encode(region_image)
    half_scale = model.encode(resize(region_image, 0.5))
    quarter_scale = model.encode(resize(region_image, 0.25))
    
    # Check if similar across scales
    sim_full_half = cosine_similarity(full_scale, half_scale)
    sim_half_quarter = cosine_similarity(half_scale, quarter_scale)
    
    # Real images: high similarity across scales
    # Compressed/manipulated: similarity drops
    if sim_full_half < threshold or sim_half_quarter < threshold:
        # Compression inconsistency detected
        suspicious_regions.add(region)
```

**Interpretation**: Different compression levels show different scale behavior.

### Method 4: Type-Specific Clustering

Check if similar region types cluster together in embedding space:

```python
# Group regions by type
photos = [r for r in regions if r.type == 'PHOTO']
texts = [r for r in regions if r.type == 'TEXT']
seals = [r for r in regions if r.type == 'SEAL']

# Within each type, check consistency
for photo in photos:
    other_photos = [p for p in photos if p != photo]
    
    # This photo should be similar to other photos
    intra_type_sim = mean([
        cosine_similarity(photo.embedding, p.embedding)
        for p in other_photos
    ])
    
    # And different from non-photos
    inter_type_sim = mean([
        cosine_similarity(photo.embedding, r.embedding)
        for r in regions if r.type != 'PHOTO'
    ])
    
    # Expect: intra > inter
    if intra_type_sim < inter_type_sim:
        # This photo doesn't cluster with other photos!
        suspicious_regions.add(photo)
```

## Decision Logic

```python
# Aggregate suspicious signals
total_regions = len(analyzed_regions)
suspicious_count = len(suspicious_regions)
suspicion_ratio = suspicious_count / total_regions

# Document verdict
if suspicion_ratio > high_threshold:  # e.g., >40%
    verdict = LIKELY_FAKE
    confidence = suspicion_ratio
elif suspicion_ratio > medium_threshold:  # e.g., >20%
    verdict = SUSPICIOUS
    confidence = suspicion_ratio
else:
    verdict = LIKELY_REAL
    confidence = 1 - suspicion_ratio
```

## Key Advantages

- **Unsupervised**: No training data needed (pre-trained models)
- **Direct Splice Detection**: Specifically targets composite forgeries
- **Robust to Content**: Works regardless of what content is
- **Interpretable**: Shows exactly which regions are inconsistent
- **No Semantic Understanding Required**: Pure visual consistency

## Limitations

- **Small Regions**: Embeddings may be noisy for tiny regions
- **Natural Variation**: Some documents may have legitimate variation
- **Model Dependency**: Quality depends on embedding model
- **Compute**: Embedding extraction can be slow for many regions
- **Threshold Sensitivity**: Consistency thresholds need tuning

## What This Detects

### Clear Signals
- **Pasted photo** from different source (different camera/lighting)
- **Overlaid text** from different rendering engine
- **Composite regions** stitched from multiple documents
- **Copy-pasted elements** from external images

### Weaker Signals
- **Compression mismatches** (different JPEG quality)
- **Resolution differences** (upscaled/downscaled content)
- **Color space differences** (RGB vs CMYK conversions)

### May Miss
- **Subtle edits** within same source image
- **Professional forgeries** maintaining visual consistency
- **Clone stamping** from within same document

## Use Cases

### Ideal For
- Detecting composite/template-based forgeries
- Documents with pasted photos
- Cases where source consistency matters
- Splice boundary identification

### Less Effective For
- Fully AI-generated documents (internally consistent)
- Simple text edits (embeddings may not capture)
- Very small/low-quality documents
- Documents with intentional style variations

## Optimization Strategies

### Region Filtering
- Skip very small regions (< 64x64 pixels)
- Prioritize high-importance regions (PHOTO, SEAL)
- Merge similar adjacent regions

### Embedding Caching
- Cache embeddings per region content hash
- Reuse across multiple analyses
- Significantly speeds up re-analysis

### Batch Processing
- Extract embeddings in batches for GPU efficiency
- Process multiple regions simultaneously

## Integration Points

Can be combined with:
- **Gemini semantic**: Embeddings for visual, Gemini for logic
- **Frequency detectors**: Embeddings as primary, frequency for confirmation
- **Traditional CV**: ELA on embedding-flagged regions
- **Human review**: Embedding inconsistency as starting point

## Cross-Modal Extensions (CLIP)

### Photo-Text Consistency
```python
# Check if passport photo matches text description
photo_emb = clip.encode_image(passport_photo)
text_emb = clip.encode_text(f"passport photo of {name}")

similarity = cosine_similarity(photo_emb, text_emb)

if similarity < threshold:
    # Photo doesn't semantically match document info
```

### Document Type Validation
```python
# Check if document matches claimed type
doc_emb = clip.encode_image(document)
type_emb = clip.encode_text(f"{country} passport")

if cosine_similarity(doc_emb, type_emb) < threshold:
    # Document doesn't look like claimed type
```

## Implementation Considerations

### Model Requirements
- DINOv2: ~300MB model, PyTorch
- CLIP: ~500MB model, PyTorch
- GPU recommended but not required

### Embedding Dimensionality
- DINOv2: 768D (base), 1024D (large)
- CLIP: 512D (ViT-B/32)
- Higher dimensions = more detail but slower

### Similarity Metric
- Cosine similarity most common
- Euclidean distance alternative
- Can normalize embeddings first

### Threshold Determination
- Start conservative (higher thresholds)
- Tune on validation set if available
- Consider adaptive thresholds per document type
