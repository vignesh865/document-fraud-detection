"""
Document Segmenter

Segments document images into semantic regions:
- TEXT: Printed or handwritten text
- PHOTO: Faces, portraits, photographs
- GRAPHIC: Logos, seals, emblems
- BACKGROUND: Uniform areas, borders
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple

# Region type constants
REGION_TEXT = 'TEXT'
REGION_PHOTO = 'PHOTO'
REGION_GRAPHIC = 'GRAPHIC'
REGION_BACKGROUND = 'BACKGROUND'


def segment_document(image_path: str) -> Dict:
    """
    Segment document into semantic regions.
    
    Args:
        image_path: Path to document image
        
    Returns:
        Dictionary with region map and region list
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            return {"error": "Could not read image"}
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        # Initialize region map (each pixel labeled with region ID)
        region_map = np.zeros((h, w), dtype=np.int32)
        regions = []
        
        # 1. Detect text regions
        text_regions = _detect_text_regions(gray)
        
        # If morphological approach failed, try OCR
        if len(text_regions) < 2:
            try:
                import pytesseract
                text_regions_ocr = _detect_text_regions_ocr(gray)
                text_regions.extend(text_regions_ocr)
            except ImportError:
                pass  # pytesseract not available
        
        for i, text_region in enumerate(text_regions):
            region_id = len(regions) + 1
            x, y, rw, rh = text_region['bbox']
            mask = text_region['mask']
            region_map[mask] = region_id
            
            regions.append({
                'id': region_id,
                'type': REGION_TEXT,
                'bbox': (x, y, rw, rh),
                'mask': mask,
                'area': np.sum(mask)
            })
        
        # 2. Detect photo regions (faces, skin tones)
        photo_regions = _detect_photo_regions(img, gray, region_map)
        for photo_region in photo_regions:
            region_id = len(regions) + 1
            x, y, rw, rh = photo_region['bbox']
            mask = photo_region['mask']
            region_map[mask] = region_id
            
            regions.append({
                'id': region_id,
                'type': REGION_PHOTO,
                'bbox': (x, y, rw, rh),
                'mask': mask,
                'area': np.sum(mask)
            })
        
        # 3. Remaining unlabeled regions = background
        background_mask = (region_map == 0)
        if np.any(background_mask):
            region_id = len(regions) + 1
            region_map[background_mask] = region_id
            
            regions.append({
                'id': region_id,
                'type': REGION_BACKGROUND,
                'bbox': (0, 0, w, h),
                'mask': background_mask,
                'area': np.sum(background_mask)
            })
        
        return {
            'region_map': region_map,
            'regions': regions,
            'num_regions': len(regions),
            'image_shape': (h, w)
        }
        
    except Exception as e:
        return {"error": str(e)}


def _detect_text_regions(gray: np.ndarray) -> List[Dict]:
    """
    Detect text regions using edge density and morphology.
    
    Text has high edge density and connected components.
    """
    # Adaptive threshold for better text detection
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY_INV, 15, 5)
    
    # Morphological operations to connect text
    # Horizontal kernel for text lines
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
    dilated = cv2.dilate(binary, kernel_h, iterations=1)
    
    # Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    h, w = gray.shape
    min_area = (h * w) * 0.002  # At least 0.2% of image (was 0.5%)
    max_area = (h * w) * 0.8    # Not more than 80% (was 70%)
    
    text_regions = []
    
    for cnt in contours:
        x, y, rw, rh = cv2.boundingRect(cnt)
        area = rw * rh
        aspect_ratio = rw / float(rh) if rh > 0 else 0
        
        # More lenient aspect ratio for text blocks
        if min_area < area < max_area and 0.8 < aspect_ratio < 40:
            # Create mask for this region
            mask = np.zeros((h, w), dtype=bool)
            cv2.drawContours(mask.astype(np.uint8), [cnt], -1, 1, -1)
            
            # Check pixel density (text should have some content)
            region_binary = binary[mask]
            pixel_density = np.sum(region_binary > 0) / area if area > 0 else 0
            
            # Text regions have moderate pixel density (0.02-0.5)
            if 0.01 < pixel_density < 0.6:  # More lenient
                text_regions.append({
                    'bbox': (x, y, rw, rh),
                    'mask': mask,
                    'pixel_density': pixel_density
                })
    
    return text_regions


def _detect_text_regions_ocr(gray: np.ndarray) -> List[Dict]:
    """
    Detect text regions using Tesseract OCR.
    
    Fallback when morphological approach fails.
    """
    try:
        import pytesseract
        
        # Get bounding boxes from OCR
        data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT)
        
        h, w = gray.shape
        min_conf = 30  # Minimum confidence
        
        # Group words into lines/blocks
        regions_dict = {}
        
        for i in range(len(data['text'])):
            if int(data['conf'][i]) > min_conf and data['text'][i].strip():
                block_num = data['block_num'][i]
                line_num = data['line_num'][i]
                key = (block_num, line_num)
                
                x, y, rw, rh = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                
                if key in regions_dict:
                    # Extend existing region
                    rx1, ry1, rx2, ry2 = regions_dict[key]
                    rx1 = min(rx1, x)
                    ry1 = min(ry1, y)
                    rx2 = max(rx2, x + rw)
                    ry2 = max(ry2, y + rh)
                    regions_dict[key] = (rx1, ry1, rx2, ry2)
                else:
                    regions_dict[key] = (x, y, x + rw, y + rh)
        
        # Convert to region format
        text_regions = []
        min_area = (h * w) * 0.001  # 0.1% minimum
        
        for (x1, y1, x2, y2) in regions_dict.values():
            rw = x2 - x1
            rh = y2 - y1
            area = rw * rh
            
            if area > min_area:
                mask = np.zeros((h, w), dtype=bool)
                mask[y1:y2, x1:x2] = True
                
                text_regions.append({
                    'bbox': (x1, y1, rw, rh),
                    'mask': mask,
                    'source': 'ocr'
                })
        
        return text_regions
        
    except Exception as e:
        return []


def _detect_photo_regions(img: np.ndarray, gray: np.ndarray, existing_map: np.ndarray) -> List[Dict]:
    """
    Detect photo/portrait regions using face detection and skin tone.
    """
    h, w = gray.shape
    photo_regions = []
    
    # Try face detection (Haar cascade)
    try:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        for (x, y, fw, fh) in faces:
            # Expand region around face (photos are usually larger than just face)
            expand = 0.3
            x1 = max(0, int(x - fw * expand))
            y1 = max(0, int(y - fh * expand))
            x2 = min(w, int(x + fw * (1 + expand)))
            y2 = min(h, int(y + fh * (1 + expand)))
            
            # Create mask
            mask = np.zeros((h, w), dtype=bool)
            mask[y1:y2, x1:x2] = True
            
            # Exclude already labeled regions
            mask &= (existing_map == 0)
            
            if np.sum(mask) > 0:
                photo_regions.append({
                    'bbox': (x1, y1, x2-x1, y2-y1),
                    'mask': mask
                })
    except Exception as e:
        # Face detection failed, continue
        pass
    
    # If no faces found, try skin tone detection (YCrCb color space)
    if len(photo_regions) == 0:
        try:
            ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
            
            # Skin tone range in YCrCb
            lower = np.array([0, 133, 77], dtype=np.uint8)
            upper = np.array([255, 173, 127], dtype=np.uint8)
            
            skin_mask = cv2.inRange(ycrcb, lower, upper)
            
            # Morphological operations
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
            skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
            skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            min_area = (h * w) * 0.02  # At least 2% of image
            
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > min_area:
                    x, y, pw, ph = cv2.boundingRect(cnt)
                    
                    # Create mask
                    mask = np.zeros((h, w), dtype=bool)
                    cv2.drawContours(mask.astype(np.uint8), [cnt], -1, 1, -1)
                    
                    # Exclude already labeled
                    mask &= (existing_map == 0)
                    
                    if np.sum(mask) > 0:
                        photo_regions.append({
                            'bbox': (x, y, pw, ph),
                            'mask': mask
                        })
        except Exception as e:
            # Skin detection failed
            pass
    
    return photo_regions


def visualize_segmentation(image_path: str, segmentation: Dict, output_path: str):
    """Generate visualization of document segmentation."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    region_map = segmentation['region_map']
    regions = segmentation['regions']
    
    # Create colored overlay
    overlay = np.zeros_like(img_rgb)
    colors = {
        REGION_TEXT: [0, 255, 0],       # Green
        REGION_PHOTO: [255, 0, 0],      # Red
        REGION_GRAPHIC: [0, 0, 255],    # Blue
        REGION_BACKGROUND: [128, 128, 128]  # Gray
    }
    
    for region in regions:
        color = colors.get(region['type'], [255, 255, 255])
        overlay[region['mask']] = color
    
    # Blend
    blended = cv2.addWeighted(img_rgb, 0.6, overlay, 0.4, 0)
    
    # Draw bounding boxes and labels
    for region in regions:
        if region['type'] != REGION_BACKGROUND:
            x, y, rw, rh = region['bbox']
            color = tuple([c//255 for c in colors[region['type']]])
            cv2.rectangle(blended, (x, y), (x+rw, y+rh), colors[region['type']], 2)
            cv2.putText(blended, region['type'], (x, y-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[region['type']], 2)
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    axes[0].imshow(img_rgb)
    axes[0].set_title('Original Document', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    axes[1].imshow(blended)
    axes[1].set_title(f'Segmentation ({len([r for r in regions if r["type"]==REGION_TEXT])} text regions, '
                     f'{len([r for r in regions if r["type"]==REGION_PHOTO])} photo regions)',
                     fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    # Legend
    legend_elements = []
    for rtype, color in colors.items():
        count = len([r for r in regions if r['type'] == rtype])
        if count > 0:
            from matplotlib.patches import Patch
            legend_elements.append(Patch(facecolor=np.array(color)/255, label=f'{rtype} ({count})'))
    
    axes[1].legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Segmentation visualization saved to {output_path}")
