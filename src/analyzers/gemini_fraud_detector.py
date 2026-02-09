"""
Gemini AI Direct Fraud Detection

Uses Gemini vision model to directly analyze documents for signs of forgery.
Much simpler than traditional CV methods and leverages AI training on billions of images.
"""

import cv2
import os
import json
import hashlib
from typing import Dict

def _compute_file_hash(file_path: str) -> str:
    """Compute SHA256 hash of file for caching."""
    sha256 = hashlib.sha256()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            sha256.update(chunk)
    return sha256.hexdigest()


def _get_cache_path(image_path: str) -> str:
    """Get cache file path for fraud analysis."""
    cache_dir = os.path.join(os.path.dirname(__file__), '.fraud_analysis_cache')
    os.makedirs(cache_dir, exist_ok=True)
    
    file_hash = _compute_file_hash(image_path)
    cache_file = os.path.join(cache_dir, f"{file_hash}.json")
    return cache_file


def _load_from_cache(image_path: str) -> Dict:
    """Load analysis from cache if exists."""
    cache_file = _get_cache_path(image_path)
    
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r') as f:
                return json.load(f)
        except Exception:
            pass
    
    return None


def _save_to_cache(image_path: str, analysis: Dict):
    """Save analysis to cache."""
    cache_file = _get_cache_path(image_path)
    
    try:
        with open(cache_file, 'w') as f:
            json.dump(analysis, f, indent=2)
    except Exception:
        pass


def analyze_with_gemini(image_path: str, api_key: str = None, use_cache: bool = True) -> Dict:
    """
    Analyze document for fraud using Gemini vision AI.
    
    Args:
        image_path: Path to document image
        api_key: Google API key (or set GEMINI_API_KEY env var)
        use_cache: Whether to use file-hash based caching
        
    Returns:
        Dictionary with fraud analysis results
    """
    # Check cache first
    if use_cache:
        cached = _load_from_cache(image_path)
        if cached:
            print(f"   📦 Using cached Gemini fraud analysis")
            cached['source'] = 'cached'
            return cached
    
    try:
        from google import genai
        
        # Configure API
        if api_key is None:
            api_key = os.getenv('GEMINI_API_KEY')
        
        if not api_key:
            return {"error": "GEMINI_API_KEY not set"}
        
        client = genai.Client(api_key=api_key)
        
        # Verify image exists
        if not os.path.exists(image_path):
            return {"error": "Image file not found"}
        
        # Upload image
        uploaded_file = client.files.upload(file=image_path)
        
        # Create forensic analysis prompt
        prompt = """You are a document forensics expert. Analyze this passport/ID document image for signs of digital forgery or tampering.

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
}"""

        # Call Gemini
        response = client.models.generate_content(
            model='gemini-2.5-pro',
            contents=[prompt, uploaded_file]
        )
        
        response_text = response.text.strip()
        
        # Extract JSON from response
        if '```json' in response_text:
            response_text = response_text.split('```json')[1].split('```')[0].strip()
        elif '```' in response_text:
            response_text = response_text.split('```')[1].split('```')[0].strip()
        
        # Parse JSON
        analysis = json.loads(response_text)
        
        # Add metadata
        analysis['source'] = 'gemini_ai'
        analysis['model'] = 'gemini-2.5-pro'
        
        # Determine verdict
        is_suspicious = analysis.get('is_forged', False)
        confidence = analysis.get('confidence', 0) / 100.0
        
        result = {
            'is_suspicious': is_suspicious,
            'confidence': confidence,
            'is_forged': analysis.get('is_forged', False),
            'red_flags': analysis.get('red_flags', []),
            'green_flags': analysis.get('green_flags', []),
            'explanation': analysis.get('explanation', ''),
            'source': 'gemini_ai',
            'model': 'gemini-2.0-flash-exp'
        }
        
        # Save to cache
        if use_cache:
            _save_to_cache(image_path, result)
        
        return result
        
    except ImportError:
        return {"error": "google-genai not installed. Run: pip install google-genai"}
    except json.JSONDecodeError as e:
        return {"error": f"Failed to parse Gemini response as JSON: {e}", "raw_response": response_text}
    except Exception as e:
        return {"error": f"Gemini analysis failed: {str(e)}"}


def visualize_analysis(image_path: str, analysis: Dict, output_path: str):
    """Create visualization of Gemini fraud analysis."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    ax.imshow(img_rgb)
    ax.axis('off')
    
    # Create verdict box
    verdict = "🚨 FORGED" if analysis.get('is_forged', False) else "✅ AUTHENTIC"
    confidence = analysis.get('confidence', 0) * 100
    
    title = f"Gemini AI Fraud Analysis\n{verdict} (Confidence: {confidence:.0f}%)"
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    
    # Add analysis text
    text_parts = []
    
    if analysis.get('red_flags'):
        text_parts.append("🚨 RED FLAGS:")
        for flag in analysis['red_flags']:
            text_parts.append(f"  • {flag}")
    
    if analysis.get('green_flags'):
        text_parts.append("\n✅ GREEN FLAGS:")
        for flag in analysis['green_flags']:
            text_parts.append(f"  • {flag}")
    
    if analysis.get('explanation'):
        text_parts.append(f"\n📝 {analysis['explanation']}")
    
    text_content = '\n'.join(text_parts)
    
    # Add text box at bottom
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.9)
    ax.text(0.5, -0.05, text_content, transform=ax.transAxes, 
            fontsize=10, verticalalignment='top', horizontalalignment='center',
            bbox=props, family='monospace')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Gemini analysis visualization saved to {output_path}")
