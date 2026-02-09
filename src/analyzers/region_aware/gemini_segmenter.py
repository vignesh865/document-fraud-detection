"""
Gemini Vision-Based Document Segmenter

Uses Gemini 2.5 Pro to detect and label document regions with bounding boxes.
Much more robust than traditional CV for passport/document analysis.
"""

import cv2
import numpy as np
import base64
import os
from typing import Dict, List
import json
import hashlib


def _compute_file_hash(file_path: str) -> str:
    """Compute SHA256 hash of file for caching."""
    sha256 = hashlib.sha256()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            sha256.update(chunk)
    return sha256.hexdigest()


def _get_cache_path(image_path: str) -> str:
    """Get cache file path for an image."""
    cache_dir = os.path.join(os.path.dirname(__file__), '.segmentation_cache')
    os.makedirs(cache_dir, exist_ok=True)
    
    file_hash = _compute_file_hash(image_path)
    cache_file = os.path.join(cache_dir, f"{file_hash}.json")
    return cache_file


def _load_from_cache(image_path: str) -> str:
    """Load raw Gemini JSON response from cache if exists."""
    cache_file = _get_cache_path(image_path)
    
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r') as f:
                return f.read()
        except Exception:
            pass
    
    return None


def _save_to_cache(image_path: str, response_text: str):
    """Save raw Gemini JSON response to cache."""
    cache_file = _get_cache_path(image_path)
    
    try:
        with open(cache_file, 'w') as f:
            f.write(response_text)
    except Exception:
        pass


def segment_document_with_gemini(image_path: str, api_key: str = None, use_cache: bool = True) -> Dict:
    """
    Segment document using Gemini 2.5 Pro vision model.
    
    Args:
        image_path: Path to document image
        api_key: Google API key (or set GEMINI_API_KEY env var)
        use_cache: Whether to use file-hash based caching (default: True)
        
    Returns:
        Dictionary with region map and region list
    """
    # Check cache first
    response_text = None
    if use_cache:
        response_text = _load_from_cache(image_path)
        if response_text:
            print(f"   📦 Using cached Gemini response")
    
    # If not in cache, call Gemini
    if response_text is None:
        try:
            from google import genai
            from google.genai import types
            
            # Configure API
            if api_key is None:
                api_key = os.getenv('GEMINI_API_KEY')
            
            if not api_key:
                return {"error": "GEMINI_API_KEY not set"}
            
            client = genai.Client(api_key=api_key)
            
            # Load image
            img = cv2.imread(image_path)
            if img is None:
                return {"error": "Could not read image"}
            
            # Upload image
            uploaded_file = client.files.upload(file=image_path)
            
            # Create prompt
            prompt = """Analyze this document image and identify ALL distinct regions.

For each region, provide:
1. Type: TEXT, PHOTO, LOGO, SEAL, BACKGROUND, or MRZ
2. Bounding box: [ymin, xmin, ymax, xmax] in normalized coordinates (0.0 to 1.0)
3. Description: Brief description of what the region contains

Return ONLY a JSON array with this structure:
[
  {
    "type": "TEXT",
    "bbox": [ymin, xmin, ymax, xmax],
    "description": "Passport number"
  },
  ...
]

Rules:
- Identify ALL text regions separately (name, number, dates, etc.)
- Detect the photo/portrait region
- Detect any logos, seals, or watermarks
- Use normalized coordinates (0.0 to 1.0 where 1.0 is image width/height)
- Return ONLY the JSON array, no other text"""
            
            # Call Gemini
            response = client.models.generate_content(
                model='gemini-2.5-pro',
                contents=[prompt, uploaded_file]
            )
            
            response_text = response.text.strip()
            
            # Save to cache
            if use_cache:
                _save_to_cache(image_path, response_text)
        
        except ImportError:
            return {"error": "google-genai not installed. Run: pip install google-genai"}
        except Exception as e:
            return {"error": f"Gemini API call failed: {str(e)}"}
    
    # Process response (same code path for cached or fresh response)
    try:
        img = cv2.imread(image_path)
        if img is None:
            return {"error": "Could not read image"}
        
        h, w = img.shape[:2]
        
        
        # Extract JSON (handle markdown code blocks)
        if '```json' in response_text:
            response_text = response_text.split('```json')[1].split('```')[0].strip()
        elif '```' in response_text:
            response_text = response_text.split('```')[1].split('```')[0].strip()
        
        regions_data = json.loads(response_text)
        
        # Convert to region format
        region_map = np.zeros((h, w), dtype=np.int32)
        regions = []
        
        for idx, region_data in enumerate(regions_data):
            region_id = idx + 1
            rtype = region_data.get('type', 'UNKNOWN')
            bbox_raw = region_data.get('bbox', [])
            
            if len(bbox_raw) != 4:
                continue
            
            # Gemini format: [ymin, xmin, ymax, xmax] in normalized coordinates (0.0-1.0)
            ymin, xmin, ymax, xmax = bbox_raw
            
            # Convert normalized to pixel coordinates
            x1 = int(xmin * w)
            y1 = int(ymin * h)
            x2 = int(xmax * w)
            y2 = int(ymax * h)
            
            # Ensure within image bounds
            x1 = max(0, min(x1, w))
            y1 = max(0, min(y1, h))
            x2 = max(0, min(x2, w))
            y2 = max(0, min(y2, h))
            
            if x2 <= x1 or y2 <= y1:
                continue
            
            # Create mask
            mask = np.zeros((h, w), dtype=bool)
            mask[y1:y2, x1:x2] = True
            
            # Update region map
            region_map[mask] = region_id
            
            regions.append({
                'id': region_id,
                'type': rtype,
                'bbox': (x1, y1, x2 - x1, y2 - y1),
                'mask': mask,
                'area': np.sum(mask),
                'description': region_data.get('description', '')
            })
        
        return {
            'region_map': region_map,
            'regions': regions,
            'num_regions': len(regions),
            'image_shape': (h, w),
            'source': 'gemini' if use_cache and response_text != _load_from_cache(image_path) else 'gemini_cached'
        }
        
    except json.JSONDecodeError as e:
        return {"error": f"Failed to parse Gemini response as JSON: {e}"}
    except Exception as e:
        return {"error": f"Processing failed: {str(e)}"}


def visualize_gemini_segmentation(image_path: str, segmentation: Dict, output_path: str):
    """Visualize Gemini-detected regions."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    regions = segmentation['regions']
    
    # Create colored overlay
    overlay = np.zeros_like(img_rgb)
    colors = {
        'TEXT': [0, 255, 0],
        'PHOTO': [255, 0, 0],
        'LOGO': [0, 0, 255],
        'SEAL': [255, 255, 0],
        'MRZ': [0, 255, 255],
        'BACKGROUND': [128, 128, 128]
    }
    
    for region in regions:
        color = colors.get(region['type'], [200, 200, 200])
        overlay[region['mask']] = color
    
    # Blend
    blended = cv2.addWeighted(img_rgb, 0.6, overlay, 0.4, 0)
    
    # Draw bounding boxes and labels
    for region in regions:
        x, y, rw, rh = region['bbox']
        color = colors.get(region['type'], [200, 200, 200])
        
        cv2.rectangle(blended, (x, y), (x+rw, y+rh), color, 2)
        
        # Label
        label = f"{region['type']}"
        if region.get('description'):
            label += f": {region['description'][:20]}"
        
        # Background for text
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(blended, (x, y-th-5), (x+tw, y), color, -1)
        cv2.putText(blended, label, (x, y-3), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, [0, 0, 0], 1)
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(18, 9))
    
    axes[0].imshow(img_rgb)
    axes[0].set_title('Original Document', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    axes[1].imshow(blended)
    
    # Count by type
    type_counts = {}
    for r in regions:
        rtype = r['type']
        type_counts[rtype] = type_counts.get(rtype, 0) + 1
    
    title = "Gemini Vision Segmentation\n"
    title += ", ".join([f"{t}: {c}" for t, c in type_counts.items()])
    
    axes[1].set_title(title, fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Gemini segmentation visualization saved to {output_path}")
