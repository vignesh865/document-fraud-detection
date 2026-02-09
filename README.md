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

We are exploring **6 distinct strategies** for document fraud detection, each with different methodological approaches:

| Strategy | Primary Method | Dependencies |
|----------|---------------|--------------|
| [Strategy 1](strategies/Strategy%201%20-%20Gemini%20AI%20Direct%20Analysis.md) | Gemini AI Direct | Gemini API |
| [Strategy 2](strategies/Strategy%202%20-%20Traditional%20Computer%20Vision.md) | Traditional CV | None (offline) |
| [Strategy 3](strategies/Strategy%203%20-%20Region-Aware%20Analysis%20with%20Gemini%20Segmentation.md) | Region + CV | Gemini API |
| [Strategy 4](strategies/Strategy%204%20-%20Region-Based%20Frequency%20Detection.md) | Region + Frequency | Gemini API |
| [Strategy 5](strategies/Strategy%205%20-%20Embedding-Based%20Region%20Consistency.md) | Embedding Consistency | DINOv2/CLIP |
| [Strategy 6](strategies/Strategy%206%20-%20Hybrid%20Multi-Tier%20System.md) | Hybrid Multi-Tier | Multiple |

---

## Strategy Summaries

### Strategy 1: Gemini AI Direct Analysis
**[Read Full Strategy →](strategies/Strategy%201%20-%20Gemini%20AI%20Direct%20Analysis.md)**

Uses Gemini Vision API for end-to-end document forensic analysis.

**Approach**: Single AI call for semantic understanding, mathematical validation (MRZ checksums, date logic), and visual inspection.



---

### Strategy 2: Traditional Computer Vision
**[Read Full Strategy →](strategies/Strategy%202%20-%20Traditional%20Computer%20Vision.md)**

Classical image processing techniques without AI/ML dependencies.

**Approach**: Apply multiple independent CV analyzers (ELA, noise analysis, font checking, frequency analysis) and aggregate results.



---

### Strategy 3: Region-Aware Analysis with Gemini Segmentation
**[Read Full Strategy →](strategies/Strategy%203%20-%20Region-Aware%20Analysis%20with%20Gemini%20Segmentation.md)**

Gemini segments document into semantic regions, then applies targeted CV analysis per region type.

**Approach**: Two-stage pipeline - Gemini identifies and classifies regions (TEXT, PHOTO, SEAL, etc.), then region-specific analyzers examine each.



---

### Strategy 4: Region-Based Frequency Detection
**[Read Full Strategy →](strategies/Strategy%204%20-%20Region-Based%20Frequency%20Detection.md)**

Applies GAN and Diffusion frequency-domain detectors to individual regions.

**Approach**: Gemini segments → Run GAN/Diffusion on each region → Aggregate suspicious signals. Includes type-aware filtering (only analyze appropriate regions).



---

### Strategy 5: Embedding-Based Region Consistency
**[Read Full Strategy →](strategies/Strategy%205%20-%20Embedding-Based%20Region%20Consistency.md)**

Uses visual embeddings (DINOv2, CLIP) to detect inconsistencies between regions.

**Approach**: Extract embeddings for each region → Compare consistency scores → Flag regions with anomalous embeddings indicating splice boundaries.



---

### Strategy 6: Hybrid Multi-Tier System
**[Read Full Strategy →](strategies/Strategy%206%20-%20Hybrid%20Multi-Tier%20System.md)**

Combines all methods in a tiered architecture prioritizing high-confidence techniques.

**Approach**: 
- **Tier 1** (Definitive): Gemini semantic checks (auto-reject if failed)
- **Tier 2** (Strong): Embeddings + splice detection (human review)
- **Tier 3** (Weak): Frequency + metadata (additional context)



---

## Supporting Documentation

### Gemini Prompts Reference
**[Read Full Reference →](strategies/Gemini%20Prompts%20Reference.md)**

Complete reference for all Gemini Vision API prompts used across strategies.

---

## Implementation Status

All strategies are documented and in exploratory phase. Current codebase includes:

- ✅ Gemini direct fraud detection (`src/analyzers/gemini_fraud_detector.py`)
- ✅ Gemini semantic segmentation (`src/analyzers/region_aware/gemini_segmenter.py`)
- ✅ Traditional CV analyzers (ELA, noise, font, PNG, etc.)
- ✅ GAN detector (`src/analyzers/gan_detector.py`)
- ✅ Diffusion detector (`src/analyzers/optimized_diffusion_detector.py`)
- ✅ Region-based GAN detection (`src/analyzers/region_gan_detector.py`)
- ✅ Region-based Diffusion detection (`src/analyzers/region_diffusion_detector.py`)
- ⏳ Embedding consistency detector (Strategy 5 - pending)
- ⏳ Multi-tier hybrid system (Strategy 6 - pending)
