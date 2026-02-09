# Document Fraud Detection Strategies

|Category            |Forgery Type         |Forensic Technique                  |Best File Formats|What it Detects                                    |How it Detects                                                                                      |
|--------------------|---------------------|------------------------------------|-----------------|---------------------------------------------------|----------------------------------------------------------------------------------------------------|
|Statistical (Pixels)|Alteration / Splicing|Error Level Analysis (ELA)          |JPEG             |Addition of text, dates, or signatures.            |Re-saves the image at a specific quality; edited areas show higher "error" (noise) levels.          |
|                    |Object Removal       |Noise Variance / Map                |PNG, BMP, RAW    |"Patching" over data (e.g., covering a name).      |Identifies local inconsistencies in the natural sensor/scanner noise of the document.               |
|                    |Cloning              |Copy-Move Detection                 |All Formats      |Duplicated signatures or background patches.       |Uses algorithms (SIFT/SURF) to find 100% identical pixel clusters that don't occur naturally.       |
|                    |Recapture            |Moiré Pattern Analysis              |All Formats      |"Re-photographing" a fake printed document.        |Scans for periodic wavy interference patterns created by digital grids/screens.                     |
|Structural (Data)   |Formatting Fraud     |Font Morphometry                    |PDF, PNG         |Replaced text or "Type-overs."                     |Measures microscopic differences in character weight, kerning (spacing), and font family.           |
|                    |PDF Tampering        |Incremental Update Analysis         |PDF              |Hidden original data under "layers."               |Analyzes the PDF "body" for previous versions of the file that weren't fully deleted.               |
|                    |Manipulation         |Bit-Plane Complexity (BPCS)         |PNG, BMP         |Hidden data or pixel modification.                 |Analyzes the least significant bits of an image; "human" edits disrupt the mathematical randomness. |
|Hardware DNA        |Source Fraud         |PRNU (Photo Response Non-Uniformity)|RAW, JPEG, PNG   |Verification of the original camera/scanner.       |Matches the "pixel fingerprint" of the sensor to prove if the document came from the claimed device.|
|                    |Screen Spoofing      |Liveness / Texture Analysis         |All Formats      |Using a high-res photo instead of a real person/ID.|Detects "flatness" or lack of 3D depth and natural light reflections on the document surface.       |
|Logical / Meta      |Origin Fraud         |EXIF/XMP Scrutiny                   |JPEG, PNG, TIFF  |Use of editing software (Photoshop, GIMP).         |Checks "Last Modified" dates and the "Creator Tool" tag in the file’s hidden header.                |
|                    |Fabrication          |MRZ Checksum Validation             |Passports, Visas |Fake ID numbers.                                   |Runs the document's numbers through a Modulo-10 formula to see if they match the security digits.   |
|Modern AI           |Deepfake/AI Gen      |GAN Artifact Detection              |All Formats      |AI-generated faces or documents.                   |Identifies "synthetic" textures or microscopic "glitches" unique to generative AI models.           |


## Experimental Strategy Overview

We've developed **6 distinct strategies** for document fraud detection, each with different strengths, limitations, and use cases:

| Strategy | Primary Method | Dependencies | Best For |
|----------|---------------|--------------|----------|
| [Strategy 1](#strategy-1) | Gemini AI Direct | Gemini API | Semantic/logical validation |
| [Strategy 2](#strategy-2) | Traditional CV | None (offline) | Privacy-critical, offline |
| [Strategy 3](#strategy-3) | Region + CV | Gemini API | Targeted analysis |
| [Strategy 4](#strategy-4) | Region + Frequency | Gemini API | AI-generated content |
| [Strategy 5](#strategy-5) | Embedding Consistency | DINOv2/CLIP | Splice detection |
| [Strategy 6](#strategy-6) | Hybrid Multi-Tier | Multiple | Production systems |

---

## Strategy Documents

### Strategy 1: Gemini AI Direct Analysis
**[Read Full Strategy →](strategies/Strategy%201%20-%20Gemini%20AI%20Direct%20Analysis.md)**

Uses Gemini Vision API for end-to-end document forensic analysis.

**Approach**: Single AI call for semantic understanding, mathematical validation (MRZ checksums, date logic), and visual inspection.

**Key Strengths**:
- Explainable AI reasoning
- Catches definitive proofs (checksum errors, impossible dates)
- No training data required

**Best For**: Quick screening, documents with semantic validation points

---

### Strategy 2: Traditional Computer Vision
**[Read Full Strategy →](strategies/Strategy%202%20-%20Traditional%20Computer%20Vision.md)**

Classical image processing techniques without AI/ML dependencies.

**Approach**: Apply multiple independent CV analyzers (ELA, noise analysis, font checking, frequency analysis) and aggregate results.

**Key Strengths**:
- Completely offline
- No API costs
- Deterministic results

**Best For**: Privacy-critical scenarios, high-volume batch processing, air-gapped environments

---

### Strategy 3: Region-Aware Analysis with Gemini Segmentation
**[Read Full Strategy →](strategies/Strategy%203%20-%20Region-Aware%20Analysis%20with%20Gemini%20Segmentation.md)**

Gemini segments document into semantic regions, then applies targeted CV analysis per region type.

**Approach**: Two-stage pipeline - Gemini identifies and classifies regions (TEXT, PHOTO, SEAL, etc.), then region-specific analyzers examine each.

**Key Strengths**:
- Reduced false positives (right tool for each region)
- Better interpretability (know which regions problematic)
- Flexible analyzer selection

**Best For**: Complex documents, mixed content types, cases needing fine-grained analysis

---

### Strategy 4: Region-Based Frequency Detection
**[Read Full Strategy →](strategies/Strategy%204%20-%20Region-Based%20Frequency%20Detection.md)**

Applies GAN and Diffusion frequency-domain detectors to individual regions.

**Approach**: Gemini segments → Run GAN/Diffusion on each region → Aggregate suspicious signals. Includes type-aware filtering (only analyze appropriate regions).

**Key Strengths**:
- Detects AI-generated content
- Granular detection (per-region verdicts)
- Complementary signals (GAN + Diffusion)

**Best For**: Documents with AI-generated photos, detecting GAN-created faces, mixed real/AI content

---

### Strategy 5: Embedding-Based Region Consistency
**[Read Full Strategy →](strategies/Strategy%205%20-%20Embedding%20Based%20Region%20Consistency.md)**

Uses visual embeddings (DINOv2, CLIP) to detect inconsistencies between regions.

**Approach**: Extract embeddings for each region → Compare consistency scores → Flag regions with anomalous embeddings indicating splice boundaries.

**Key Strengths**:
- Unsupervised (no training data)
- Direct splice detection
- Detects composite forgeries

**Best For**: Template-based forgeries, pasted photos, documents with content from multiple sources

---

### Strategy 6: Hybrid Multi-Tier System
**[Read Full Strategy →](strategies/Strategy%206%20-%20Hybrid%20Multi-Tier%20System.md)**

Combines all methods in a tiered architecture prioritizing high-confidence techniques.

**Approach**: 
- **Tier 1** (Definitive): Gemini semantic checks (auto-reject if failed)
- **Tier 2** (Strong): Embeddings + splice detection (human review)
- **Tier 3** (Weak): Frequency + metadata (additional context)

**Key Strengths**:
- Optimal resource usage
- Graceful degradation
- Explainable decisions

**Best For**: Production fraud detection systems, cases requiring accuracy/throughput balance

---

## Supporting Documentation

### Gemini Prompts Reference
**[Read Full Reference →](strategies/Gemini%20Prompts%20Reference.md)**

Complete reference for all Gemini Vision API prompts used across strategies.

**Contents**:
- Forensic Analysis Prompt (full text + schema)
- Semantic Segmentation Prompt (full text + schema)
- Usage examples and response parsing
- Caching strategies and cost optimization

---

## Quick Selection Guide

**Need definitive proofs with zero false positives?**  
→ Use [Strategy 1](#strategy-1) or [Strategy 6 Tier 1](#strategy-6)

**Need to work offline without API access?**  
→ Use [Strategy 2](#strategy-2)

**Have composite forgeries (pasted photos/text)?**  
→ Use [Strategy 5](#strategy-5) or [Strategy 3](#strategy-3)

**Suspect AI-generated content?**  
→ Use [Strategy 4](#strategy-4) with type filtering

**Building production system?**  
→ Use [Strategy 6](#strategy-6) for comprehensive coverage

---

## Implementation Status

All strategies are documented and ready for implementation. Current codebase includes:

- ✅ Gemini direct fraud detection (`src/analyzers/gemini_fraud_detector.py`)
- ✅ Gemini semantic segmentation (`src/analyzers/region_aware/gemini_segmenter.py`)
- ✅ Traditional CV analyzers (ELA, noise, font, PNG, etc.)
- ✅ GAN detector (`src/analyzers/gan_detector.py`)
- ✅ Diffusion detector (`src/analyzers/optimized_diffusion_detector.py`)
- ✅ Region-based GAN detection (`src/analyzers/region_gan_detector.py`)
- ✅ Region-based Diffusion detection (`src/analyzers/region_diffusion_detector.py`)
- ⏳ Embedding consistency detector (Strategy 5 - pending)
- ⏳ Multi-tier hybrid system (Strategy 6 - pending)

---

## Next Steps

1. **Choose strategy** based on use case and requirements
2. **Review strategy document** for implementation details
3. **Check Gemini Prompts Reference** if using Gemini-based strategies
4. **Implement and test** on sample documents
5. **Tune thresholds** based on validation results

---

## Contributing

When adding new strategies:
1. Create new `Strategy X - [Name].md` file
2. Follow existing format (Overview, Approach, Advantages, Limitations, etc.)
3. Update this README with new strategy entry
4. Add to Quick Selection Guide if applicable
