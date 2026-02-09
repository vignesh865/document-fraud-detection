# Gemini Prompts Reference

This document contains all Gemini Vision API prompts used in the document fraud detection strategies.

---

## 1. Forensic Analysis Prompt

**Used in**: Strategy 1 (Gemini AI Direct Analysis), Strategy 6 Tier 1 (Hybrid Multi-Tier)

**Purpose**: Analyze entire document for signs of digital forgery or tampering

**File**: `src/analyzers/gemini_fraud_detector.py`

**Model**: `gemini-2.5-pro`

### Full Prompt

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

### Expected Response Schema

```json
{
  "is_forged": boolean,
  "confidence": number (0-100),
  "red_flags": string[],
  "green_flags": string[],
  "explanation": string
}
```

### Usage Example

```python
from google import genai

client = genai.Client(api_key=api_key)
uploaded_file = client.files.upload(file=image_path)

response = client.models.generate_content(
    model='gemini-2.5-pro',
    contents=[prompt, uploaded_file]
)

analysis = json.loads(response.text)
```

### Key Features

- **Forensic Focus**: Emphasizes technical signs over general validity
- **Structured Areas**: Breaks analysis into 5 key aspects
- **JSON Only**: Explicitly requests structured JSON response
- **Binary Verdict**: Clear `is_forged` boolean decision
- **Confidence Score**: Numerical confidence (0-100)
- **Dual Flags**: Both red flags (suspicious) and green flags (authentic)
- **Explanation**: Human-readable summary

### Common Response Patterns

**For Forged Documents**:
```json
{
  "is_forged": true,
  "confidence": 85,
  "red_flags": [
    "Inconsistent font rendering in name field",
    "Photo has different lighting than document background",
    "MRZ checksum validation failed"
  ],
  "green_flags": [],
  "explanation": "Multiple technical indicators suggest digital manipulation, particularly in the name and photo regions."
}
```

**For Authentic Documents**:
```json
{
  "is_forged": false,
  "confidence": 75,
  "red_flags": [],
  "green_flags": [
    "Consistent text rendering throughout",
    "Natural document aging and wear patterns",
    "Security features appear genuine"
  ],
  "explanation": "Document shows expected characteristics of genuine passport with no signs of digital manipulation."
}
```

---

## 2. Semantic Segmentation Prompt

**Used in**: Strategy 3 (Region-Aware Analysis), Strategy 4 (Frequency Detection), Strategy 5 (Embedding Consistency), Strategy 6 Tier 2

**Purpose**: Segment document into semantic regions with type classification

**File**: `src/analyzers/region_aware/gemini_segmenter.py`

**Model**: `gemini-2.0-flash-exp`

### Full Prompt

```
Analyze this document image and identify ALL distinct regions.

For each region, provide:
1. A unique ID number
2. The TYPE from: TEXT, PHOTO, SEAL, LOGO, WATERMARK, BARCODE, MRZ (machine readable zone), BACKGROUND, SIGNATURE, or OTHER
3. Bounding box coordinates [x, y, width, height] in pixels
4. A brief description of what the region contains

Be very thorough - identify every text block, photo, seal, logo, and other elements separately.

Respond with ONLY valid JSON (no markdown):
{
  "regions": [
    {
      "id": 1,
      "type": "PHOTO",
      "bbox": [x, y, width, height],
      "description": "Passport photo of the holder"
    },
    {
      "id": 2,
      "type": "TEXT",
      "bbox": [x, y, width, height],
      "description": "Surname field"
    }
  ]
}

Make sure to identify:
- Individual text fields (name, DOB, number, etc.)
- Photos (main portrait, ghost image if present)
- Seals and watermarks
- MRZ (machine readable zone at bottom)
- Logos and emblems
- Background regions
- Any signatures
```

### Expected Response Schema

```json
{
  "regions": [
    {
      "id": number,
      "type": "TEXT" | "PHOTO" | "SEAL" | "LOGO" | "WATERMARK" | "BARCODE" | "MRZ" | "BACKGROUND" | "SIGNATURE" | "OTHER",
      "bbox": [x, y, width, height],
      "description": string
    }
  ]
}
```

### Usage Example

```python
from google import genai

client = genai.Client(api_key=api_key)
uploaded_file = client.files.upload(file=image_path)

response = client.models.generate_content(
    model='gemini-2.0-flash-exp',
    contents=[prompt, uploaded_file]
)

segmentation = json.loads(response.text)
regions = segmentation['regions']
```

### Key Features

- **Comprehensive**: Identifies ALL distinct regions
- **Type Classification**: Categorizes each region semantically
- **Bounding Boxes**: Provides pixel coordinates for extraction
- **Descriptions**: Human-readable content description
- **Specific Guidance**: Lists what to look for (prevents omissions)

### Region Types

| Type | Description | Examples |
|------|-------------|----------|
| `TEXT` | Text content | Name, DOB, document number, address |
| `PHOTO` | Photographs | Passport portrait, ghost image |
| `SEAL` | Official seals | Government seal, embossing |
| `LOGO` | Logos/emblems | Country emblem, issuing authority |
| `WATERMARK` | Watermarks | Security watermarks, background patterns |
| `BARCODE` | Barcodes/QR | 1D barcode, 2D QR code, PDF417 |
| `MRZ` | Machine readable zone | 2-3 lines of OCR-readable text at bottom |
| `BACKGROUND` | Background areas | Plain areas, decorative patterns |
| `SIGNATURE` | Signatures | Holder signature, official signature |
| `OTHER` | Uncategorized | Anything not fitting above |

### Common Response Pattern

```json
{
  "regions": [
    {
      "id": 1,
      "type": "LOGO",
      "bbox": [50, 30, 100, 100],
      "description": "National emblem at top left"
    },
    {
      "id": 2,
      "type": "PHOTO",
      "bbox": [500, 200, 300, 400],
      "description": "Portrait of the passport holder"
    },
    {
      "id": 3,
      "type": "TEXT",
      "bbox": [150, 250, 250, 40],
      "description": "Full name: SMITH, JOHN"
    },
    {
      "id": 4,
      "type": "TEXT",
      "bbox": [150, 300, 250, 40],
      "description": "Date of birth: 15 JAN 1990"
    },
    {
      "id": 5,
      "type": "MRZ",
      "bbox": [100, 950, 800, 80],
      "description": "Machine readable zone with document number and checksums"
    },
    {
      "id": 6,
      "type": "SEAL",
      "bbox": [650, 650, 120, 120],
      "description": "Official seal of the issuing authority"
    }
  ]
}
```

### Typical Region Count

- **Passports**: 30-44 regions
- **IDs**: 15-25 regions
- **Receipts**: 20-40 regions

---

## 3. Combined Usage in Strategies

### Strategy 1: Direct Analysis Only

```python
# Single Gemini call for complete analysis
result = analyze_with_gemini(image_path, api_key)

if result['is_forged']:
    print(f"FAKE: {result['explanation']}")
    print(f"Red flags: {result['red_flags']}")
```

### Strategy 3: Segmentation + Per-Region Analysis

```python
# Step 1: Segment with Gemini
segmentation = segment_document_with_gemini(image_path, api_key)

# Step 2: Analyze each region with CV
for region in segmentation['regions']:
    if region['type'] == 'PHOTO':
        analyze_photo(region)
    elif region['type'] == 'TEXT':
        analyze_text(region)
    elif region['type'] == 'MRZ':
        validate_mrz(region)
```

### Strategy 6: Hybrid (Both Prompts)

```python
# Tier 1: Use forensic analysis for definitive proofs
forensic = analyze_with_gemini(image_path, api_key)

if forensic['is_forged'] and forensic['confidence'] > 90:
    return "DEFINITIVE_FAKE"

# Tier 2: Use segmentation for targeted analysis
segmentation = segment_document_with_gemini(image_path, api_key)

for region in segmentation['regions']:
    # Apply embedding/CV analysis per region type
    analyze_region(region)
```

---

## 4. Prompt Engineering Tips

### For Forensic Analysis

**Do**:
- Emphasize "technical" and "forensic" analysis
- Request specific details, not general impressions
- Specify JSON format explicitly
- List concrete areas to examine

**Don't**:
- Ask about document validity (legal issues)
- Request speculation without evidence
- Allow free-form text responses
- Mix multiple intents in one prompt

### For Segmentation

**Do**:
- Request "ALL distinct regions" (comprehensive)
- Provide concrete type taxonomy
- List specific things to look for
- Request structured JSON

**Don't**:
- Allow vague region descriptions
- Permit arbitrary type names
- Skip bounding box requirements
- Allow nested or overlapping regions

---

## 5. Response Parsing

### Handling Markdown Wrappers

Gemini sometimes wraps JSON in markdown code blocks:

```python
response_text = response.text.strip()

# Remove markdown code blocks if present
if '```json' in response_text:
    response_text = response_text.split('```json')[1].split('```')[0].strip()
elif '```' in response_text:
    response_text = response_text.split('```')[1].split('```')[0].strip()

# Parse JSON
result = json.loads(response_text)
```

### Error Handling

```python
try:
    analysis = json.loads(response_text)
except json.JSONDecodeError as e:
    return {
        "error": f"Failed to parse Gemini response: {e}",
        "raw_response": response_text
    }
```

---


