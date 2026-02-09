# Strategy 6: Hybrid Multi-Tier Detection System

## Overview

Combine multiple detection strategies in a tiered architecture that prioritizes high-confidence methods and escalates ambiguous cases. Balances accuracy, cost, and speed by applying appropriate techniques at each tier.

## Core Philosophy

**Not all detection methods are equal**:
- Some provide definitive proofs (MRZ checksums)
- Some provide strong indicators (splice detection)  
- Some provide weak signals (frequency anomalies)

**Tier system**:
- Apply definitive proofs first (fast, free, high confidence)
- Apply strong indicators to unclear cases (moderate cost/time)
- Apply weak signals for additional context (expensive, slow)
- Escalate to human review when needed

## Three-Tier Architecture

### Tier 1: Definitive Proofs (Auto-Reject)

**Goal**: Catch mathematical/logical impossibilities with ZERO false positives

**Methods**:
- **MRZ Checksum Validation** (Gemini AI)
  - Validate machine-readable zone checksums
  - Cross-check MRZ against visual text
  - Auto-reject if checksum fails

- **Date Logic Validation** (Gemini AI or rules)
  - Check for future dates
  - Validate age consistency (DOB vs issue date)
  - Check expiration logic
  - Auto-reject if logic fails

- **Document Structure Validation** (Gemini AI)
  - Verify required fields present
  - Check format matches document type
  - Validate country-specific requirements

**Decision**:
```python
tier1_result = run_tier1_checks(document)

if tier1_result.has_definitive_proof_of_forgery:
    return {
        'verdict': 'DEFINITIVE_FAKE',
        'confidence': 100,
        'reason': tier1_result.failure_reason,
        'action': 'AUTO_REJECT'
    }
else:
    # Proceed to Tier 2
```

**Characteristics**:
- **Speed**: Fast (<1 second)
- **Cost**: Low (single Gemini call or rules)
- **False Positives**: 0%
- **Coverage**: 30-50% of obvious fakes

---

### Tier 2: Strong Indicators (Human Review)

**Goal**: Detect composite forgeries with high confidence but potential false positives

**Methods**:

**2A. Embedding Consistency** (Primary)
- Extract DINOv2 embeddings per region
- Compute region consistency scores
- Identify splice boundaries
- Flag: Regions with low consistency

**2B. Splice Detection** (Secondary)
- Error Level Analysis (ELA)
- Compression boundary detection
- Resolution inconsistency analysis
- Flag: Compression/quality mismatches

**2C. Font/Text Analysis** (Tertiary)
- Font consistency checking
- Anti-aliasing analysis
- Text alignment validation
- Flag: Text overlay or substitution

**Decision**:
```python
tier2_results = run_tier2_checks(document)

suspicious_signals = (
    tier2_results.embedding_inconsistencies +
    tier2_results.splice_detections +
    tier2_results.font_anomalies
)

if suspicious_signals >= high_threshold:
    return {
        'verdict': 'LIKELY_FAKE',
        'confidence': 70-90,
        'evidence': tier2_results,
        'action': 'HUMAN_REVIEW_RECOMMENDED'
    }
elif suspicious_signals >= medium_threshold:
    # Proceed to Tier 3 for more evidence
else:
    return {
        'verdict': 'LIKELY_REAL',
        'confidence': 70-90,
        'action': 'ACCEPT'
    }
```

**Characteristics**:
- **Speed**: Moderate (5-15 seconds)
- **Cost**: Moderate (embeddings + CV analysis)
- **False Positives**: <10%
- **Coverage**: Additional 30-40% of fakes

---

### Tier 3: Weak Signals (Contextual Evidence)

**Goal**: Provide additional context for ambiguous cases

**Methods**:

**3A. Type-Aware Frequency Analysis**
- GAN detection on PHOTO regions only
- Diffusion detection on appropriate regions
- Provide technical evidence for reviewers

**3B. Metadata Analysis**
- EXIF examination
- Edit history review
- Software fingerprinting

**3C. Cross-Modal Validation** (CLIP)
- Photo-text consistency
- Document type matching
- Semantic alignment

**Decision**:
```python
tier3_results = run_tier3_checks(document)

# Combine with Tier 2 evidence
all_evidence = {
    'tier2': tier2_results,
    'tier3': tier3_results
}

# Provide to human reviewer
return {
    'verdict': 'UNCERTAIN',
    'confidence': 50-70,
    'evidence': all_evidence,
    'action': 'HUMAN_REVIEW_REQUIRED'
}
```

**Characteristics**:
- **Speed**: Slow (30-60 seconds)
- **Cost**: High (multiple detectors)
- **False Positives**: 10-30%
- **Coverage**: Context for remaining 20-30%

---

## Workflow

```
Document Input
      ↓
┌───────────────────────────────────┐
│ TIER 1: Definitive Proofs         │
│ - MRZ Checksums                   │
│ - Date Logic                      │
│ - Structure Validation            │
└───────────────────────────────────┘
      ↓
   Definitive     → AUTO-REJECT (30-50% of fakes)
   Proof Found?
      ↓ No
┌───────────────────────────────────┐
│ TIER 2: Strong Indicators         │
│ - Embedding Consistency           │
│ - Splice Detection                │
│ - Font Analysis                   │
└───────────────────────────────────┘
      ↓
   High Signal?  → LIKELY FAKE      → Human Review (30-40% of fakes)
      ↓ No
   Low Signal?   → LIKELY REAL      → Accept (most real docs)
      ↓ Uncertain
┌───────────────────────────────────┐
│ TIER 3: Weak Signals              │
│ - Frequency Analysis              │
│ - Metadata                        │
│ - Cross-Modal                     │
└───────────────────────────────────┘
      ↓
   Additional Context → Human Review (remaining 20-30%)
```

## Integration Strategy

### Parallel Execution Where Possible
```python
# Tier 1: Sequential (fast, gates rest)
tier1 = run_tier1(document)
if tier1.is_definitive_fake:
    return reject(tier1)

# Tier 2: Parallel (independent methods)
with ThreadPoolExecutor() as executor:
    embedding_future = executor.submit(embedding_analysis, document)
    splice_future = executor.submit(splice_detection, document)
    font_future = executor.submit(font_analysis, document)
    
    embedding_result = embedding_future.result()
    splice_result = splice_future.result()
    font_result = font_future.result()

# Tier 3: Parallel (only if needed)
if is_uncertain(tier2_results):
    with ThreadPoolExecutor() as executor:
        frequency_future = executor.submit(frequency_analysis, document)
        metadata_future = executor.submit(metadata_analysis, document)
        # ...
```

### Caching Strategy
- Cache Gemini segmentation (used by multiple tiers)
- Cache embeddings (reused in Tier 2 and 3)
- Cache region extractions
- Use document hash as cache key

### Early Termination
```python
if tier1.is_definitive:
    return tier1.verdict  # Skip Tiers 2 & 3

if tier2.confidence > 90:
    return tier2.verdict  # Skip Tier 3

# Only run Tier 3 if truly uncertain
```

## Decision Matrix

| Tier 1 | Tier 2 | Tier 3 | Final Verdict | Action |
|--------|--------|--------|---------------|--------|
| FAIL | - | - | DEFINITIVE_FAKE | Auto-Reject |
| PASS | HIGH | - | LIKELY_FAKE | Human Review |
| PASS | LOW | - | LIKELY_REAL | Accept |
| PASS | MED | SUPPORTIVE | LIKELY_FAKE | Human Review |
| PASS | MED | CONTRADICTORY | UNCERTAIN | Human Review |

## Key Advantages

- **Optimal Resource Usage**: Expensive methods only when needed
- **Fast Common Cases**: Most clear cases resolved at Tier 1
- **Graceful Degradation**: Works even if some tiers fail
- **Explainable**: Clear evidence trail for decisions
- **Tunable**: Adjust thresholds per use case
- **Scalable**: Parallelize within tiers

## Limitations

- **Complexity**: More complex to implement and maintain
- **Threshold Tuning**: Multiple thresholds to configure
- **Partial Failures**: Must handle tier failures gracefully
- **Latency Variability**: Time varies by tier reached

## Use Cases

### Ideal For
- Production fraud detection systems
- Cases requiring balance of accuracy and throughput
- Scenarios with mix of obvious and subtle fakes
- Systems needing explainable decisions
- Cost-sensitive high-volume applications

### Configuration Per Use Case

**High Security** (e.g., border control):
- Tier 1: Standard
- Tier 2: Low thresholds (more to review)
- Tier 3: Always run
- Action: Route most to human review

**High Volume** (e.g., online verification):
- Tier 1: Aggressive (reject obvious)
- Tier 2: High thresholds (fewer false flags)
- Tier 3: Skip or sample
- Action: Minimize human review

**Forensic Analysis** (e.g., investigation):
- Tier 1: Standard
- Tier 2: Always run
- Tier 3: Always run
- Action: Provide all evidence to investigator

## Implementation Considerations

### Error Handling
- Each tier must handle failures independently
- Fall back to next tier if current tier errors
- Always provide *some* verdict (never crash)

### Monitoring
- Track which tier resolved each document
- Monitor tier-specific accuracy
- Identify tier bottlenecks
- Optimize slow tiers

### Continuous Improvement
- Collect human review outcomes
- Retrain thresholds based on feedback
- Add new methods to appropriate tiers
- Remove underperforming methods

### Human Reviewer Interface
- Show evidence from all executed tiers
- Highlight strongest signals
- Provide visualizations (heatmaps, highlights)
- Allow reviewer override
- Collect reviewer rationale for learning
