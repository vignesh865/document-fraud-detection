"""
Region-Based Diffusion Detection

Combines Gemini vision segmentation with Diffusion detection.
Analyzes EACH REGION separately to identify AI-generated content.

Theory: AI-generated regions have different spectral characteristics.
"""

import cv2
import numpy as np
import os
from typing import Dict, List
from src.analyzers.region_aware.gemini_segmenter import segment_document_with_gemini
from src.analyzers.optimized_diffusion_detector import OptimizedDiffusionDetector

def analyze_document_regions_with_diffusion(image_path: str, api_key: str = None) -> Dict:
    """
    Analyze each document region separately with Diffusion detection.
    
    Args:
        image_path: Path to document image
        api_key: Gemini API key
        
    Returns:
        Dictionary with analysis results
    """
    # Step 1: Segment document with Gemini
    print("   🔍 Segmenting document with Gemini...")
    segmentation = segment_document_with_gemini(image_path, api_key=api_key)
    
    if 'error' in segmentation:
        return {"error": segmentation['error']}
    
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        return {"error": "Could not load image"}
    
    regions = segmentation.get('regions', [])
    print(f"   ✅ Found {len(regions)} regions")
    
    # Step 2: Analyze each region with Diffusion detector
    print("   🤖 Running Diffusion detection on each region...")
    
    # Initialize detector once
    diffusion_detector = OptimizedDiffusionDetector()
    
    region_results = []
    suspicious_regions = []
    
    for region in regions:
        region_id = region['id']
        region_type = region['type']
        x, y, w, h = region['bbox']
        
        # Extract region
        region_img = img[y:y+h, x:x+w]
        
        # Skip tiny regions (detector needs reasonable size)
        if w < 50 or h < 50:
            continue
        
        # Save region temporarily
        temp_path = f"/tmp/region_{region_id}.png"
        cv2.imwrite(temp_path, region_img)
        
        # Run Diffusion detection
        try:
            # analyze_image returns results dict with nested 'verdict' dict
            diffusion_result = diffusion_detector.analyze_image(temp_path)
            
            # verdict info is nested in results['verdict']
            verdict_info = diffusion_result.get('verdict', {})
            verdict = verdict_info.get('verdict', 'LIKELY_REAL')
            confidence = verdict_info.get('confidence', 0)
            suspicious_signals = verdict_info.get('suspicious_signals', 0)
            
            # Debug: Print first 3 regions
            if len(region_results) < 3:
                print(f"      Region {region_id} ({region_type}):")
                print(f"         Verdict: {verdict}")
                print(f"         Signals: {suspicious_signals}/5")
                print(f"         Confidence: {confidence:.1%}")
            
            # Consider LIKELY_DIFFUSION_MODEL or SUSPICIOUS as AI-generated
            is_ai = verdict in ['LIKELY_DIFFUSION_MODEL', 'SUSPICIOUS']
            
            region_results.append({
                'id': region_id,
                'type': region_type,
                'bbox': (x, y, w, h),
                'description': region.get('description', ''),
                'diffusion_verdict': verdict,
                'diffusion_confidence': confidence,
                'suspicious_signals': suspicious_signals,
                'is_ai': is_ai
            })
            
            if is_ai:
                suspicious_regions.append({
                    'id': region_id,
                    'type': region_type,
                    'description': region.get('description', ''),
                    'confidence': confidence
                })
            
            # Cleanup
            os.remove(temp_path)
            
        except Exception as e:
            print(f"      ⚠️  Diffusion detection failed for region {region_id}: {e}")
            continue
    
    # Step 3: Analyze results
    total_analyzed = len(region_results)
    num_suspicious = len(suspicious_regions)
    
    if total_analyzed == 0:
        return {"error": "No regions could be analyzed"}
    
    suspicion_ratio = num_suspicious / total_analyzed
    
    # Decision logic
    is_suspicious = suspicion_ratio > 0.3  # If >30% of regions flagged
    
    return {
        'is_suspicious': is_suspicious,
        'total_regions': len(regions),
        'analyzed_regions': total_analyzed,
        'suspicious_regions_count': num_suspicious,
        'suspicion_ratio': suspicion_ratio,
        'suspicious_regions': suspicious_regions,
        'all_region_results': region_results,
        'source': 'diffusion_region_detection'
    }


def visualize_diffusion_regions(image_path: str, analysis: Dict, output_path: str):
    """Visualize Diffusion detection results on regions."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.imshow(img_rgb)
    
    # Draw regions
    for region_result in analysis.get('all_region_results', []):
        x, y, w, h = region_result['bbox']
        is_ai = region_result['is_ai']
        verdict = region_result.get('diffusion_verdict', 'UNKNOWN')
        signals = region_result.get('suspicious_signals', 0)
        
        # Color based on result
        if is_ai:
            color = 'red'
            label = f"🚨 {region_result['type']}\n{verdict} ({signals}/5)"
        else:
            color = 'green'
            label = f"✅ {region_result['type']}\n{verdict} ({signals}/5)"
        
        # Draw bounding box
        rect = patches.Rectangle((x, y), w, h, linewidth=2, 
                                 edgecolor=color, facecolor='none')
        ax.add_patch(rect)
        
        # Add label
        ax.text(x, y-5, label, fontsize=8, color=color, 
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    ax.axis('off')
    
    # Title
    is_suspicious = analysis.get('is_suspicious', False)
    ratio = analysis.get('suspicion_ratio', 0) * 100
    title = f"Diffusion Region Analysis - {'🚨 SUSPICIOUS' if is_suspicious else '✅ CLEAN'}\n"
    title += f"{analysis['suspicious_regions_count']}/{analysis['analyzed_regions']} regions flagged ({ratio:.0f}%)"
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Diffusion region visualization saved to {output_path}")
