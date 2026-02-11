# Strategy 1: Gemini AI Direct Forensic Analysis

## Approach

This strategy leverages the Gemini Vision API to perform a holistic forensic analysis of the document image. We treat the AI model as a specialized document forensics expert, instructing it to examine the image for specific signs of digital manipulation.

The analysis focuses on three core areas:

1.  **Mathematical and Logical Consistency**: The model explicitly verifies that machine-readable zones (MRZ) checksums are mathematically valid and match the visual text. It also checks for logical impossibilities, such as issue dates being after expiration dates or birth dates that don't match the apparent age or other data points.

2.  **Visual Artifacts**: The model scans for visual anomalies that suggest editing, such as inconsistent font rendering, "halos" around text or photos indicating copy-pasting, and localized blurring or pixelation differences.

3.  **Structural Validity**: The system checks if the document's layout, security features (like watermarks and seals), and overall structure align with the expected standards for that document type.

By relying on the model's semantic understanding, this approach aims to catch "logical" forgeries—documents that visually look perfect but contain contradictory or mathematically impossible data—which traditional computer vision often misses.

## Forensic Analysis Prompt

The following prompt is used to instruct the Gemini model:

```
You are a document forensics expert. Analyze this passport/ID document image for signs of digital forgery or tampering.

**Examine carefully:**

1. **Text Quality & Consistency**
   - Are fonts consistent throughout?
   - Any differences in text rendering quality, sharpness, or aliasing?
   - Do character sizes/styles look natural for this document type?

2. **Visual Artifacts**
   - Any visible copy-paste boundaries or cloning artifacts?
   - Unnatural edges or halos around text/photos?
   - Inconsistent pixelation or compression between regions?

3. **Photo Analysis**
   - Does the portrait photo look authentic or manipulated?
   - Any signs of face manipulation or photo replacement?
   - Natural lighting and shadows?

4. **Document Structure**
   - Do security features (watermarks, seals) look authentic?
   - Is the overall layout consistent with genuine documents?
   - Any signs of digital composition (layering)?

5. **Color & Lighting**
   - Consistent color temperature throughout?
   - Natural lighting across the document?
   - Any color discontinuities suggesting editing?

**IMPORTANT**: Be objective and forensic. Look for actual technical signs of manipulation, not just document validity.

Respond with ONLY valid JSON (no markdown, no explanations outside JSON):
{
  "is_forged": true or false,
  "confidence": 0-100 (your confidence in the verdict),
  "red_flags": [
    "Specific technical issue 1",
    "Specific technical issue 2"
  ],
  "green_flags": [
    "Positive authenticity indicator 1"
  ],
  "explanation": "Brief technical summary of your analysis"
}
```
