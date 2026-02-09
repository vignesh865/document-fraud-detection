"""
Region-Aware Document Analyzer (Orchestrator)

Coordinates the entire region-aware fraud detection pipeline:
1. Segment document into regions
2. Analyze text regions (compare text-to-text only)
3. Analyze boundaries (detect pasting)
4. Aggregate results
"""

import cv2
import numpy as np
from typing import Dict
from .segmenter import segment_document, visualize_segmentation
from .text_analyzer import analyze_text_regions
from .boundary_analyzer import analyze_boundaries


def analyze_document(image_path: str, output_path: str = None) -> Dict:
    """
    Perform region-aware fraud detection on a document.
    
    Args:
        image_path: Path to document image
        output_path: Optional path to save visualization
        
    Returns:
        Comprehensive analysis results
    """
    try:
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            return {"error": "Could not read image"}
        
        # Step 1: Segment document
        # Try Gemini first (much more accurate), fallback to CV
        from .gemini_segmenter import segment_document_with_gemini
        
        segmentation = segment_document_with_gemini(image_path)
        
        if 'error' in segmentation:
            # Fallback to traditional CV segmentation
            print(f"   ℹ️  Gemini segmentation unavailable ({segmentation['error']}), using CV fallback")
            segmentation = segment_document(image_path)
        else:
            print(f"   ✅ Using Gemini vision segmentation")
        
        if 'error' in segmentation:
            return segmentation
        
        regions = segmentation['regions']
        region_map = segmentation['region_map']
        
        # Step 2: Analyze text regions (text-to-text comparison)
        text_analysis = analyze_text_regions(regions, img)
        
        # Step 3: Analyze boundaries (paste detection)
        boundary_analysis = analyze_boundaries(region_map, img)
        
        # Step 4: Aggregate results
        suspicious_signals = 0
        reasons = []
        
        if text_analysis.get('is_suspicious', False):
            suspicious_signals += 1
            reasons.append("Text region inconsistency")
        
        if boundary_analysis.get('is_suspicious', False):
            suspicious_signals += 1
            reasons.append("Sharp paste boundaries detected")
        
        # Overall verdict
        # Need at least 1 suspicious signal from region-aware tests
        is_suspicious = suspicious_signals >= 1
        
        # Confidence score
        confidence = suspicious_signals / 2.0  # Max 2 signals
        
        results = {
            'is_suspicious': bool(is_suspicious),
            'confidence': float(confidence),
            'suspicious_signals': suspicious_signals,
            'total_checks': 2,
            'reasons': reasons,
            
            # Sub-results
            'segmentation': {
                'num_regions': segmentation['num_regions'],
                'num_text_regions': len([r for r in regions if r['type'] == 'TEXT']),
                'num_photo_regions': len([r for r in regions if r['type'] == 'PHOTO']),
            },
            'text_analysis': text_analysis,
            'boundary_analysis': boundary_analysis,
            
            'interpretation': 'Region-aware analysis: compares only like-to-like regions to avoid false positives'
        }
        
        # Visualization
        if output_path:
            _visualize_results(image_path, segmentation, text_analysis, 
                              boundary_analysis, results, output_path)
        
        return results
        
    except Exception as e:
        return {"error": str(e)}


def _visualize_results(image_path: str, segmentation: Dict, 
                       text_analysis: Dict, boundary_analysis: Dict,
                       results: Dict, output_path: str):
    """Generate comprehensive visualization."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from .segmenter import REGION_TEXT, REGION_PHOTO, REGION_BACKGROUND
    
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    region_map = segmentation['region_map']
    regions = segmentation['regions']
    
    # Create visualizations
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # 1. Original
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(img_rgb)
    ax1.set_title('Original Document', fontsize=12, fontweight='bold')
    ax1.axis('off')
    
    # 2. Segmentation overlay
    ax2 = fig.add_subplot(gs[0, 1])
    overlay = np.zeros_like(img_rgb)
    colors = {
        REGION_TEXT: [0, 255, 0],
        REGION_PHOTO: [255, 0, 0],
        REGION_BACKGROUND: [100, 100, 100]
    }
    
    for region in regions:
        color = colors.get(region['type'], [255, 255, 255])
        overlay[region['mask']] = color
    
    blended = cv2.addWeighted(img_rgb, 0.5, overlay, 0.5, 0)
    
    # Draw boxes for text regions
    for region in regions:
        if region['type'] == REGION_TEXT:
            x, y, w, h = region['bbox']
            cv2.rectangle(blended, (x, y), (x+w, y+h), [0, 255, 0], 2)
    
    ax2.imshow(blended)
    ax2.set_title(f'Segmentation\n({segmentation["num_regions"]} regions)', 
                  fontsize=12, fontweight='bold')
    ax2.axis('off')
    
    # 3. Text analysis summary
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.axis('off')
    
    text_summary = "TEXT ANALYSIS\n\n"
    if 'num_text_regions' in text_analysis:
        text_summary += f"Text Regions: {text_analysis['num_text_regions']}\n\n"
        
        if text_analysis['num_text_regions'] >= 2:
            text_summary += f"Stroke CV: {text_analysis.get('stroke_cv', 0):.3f}\n"
            text_summary += f"Char CV: {text_analysis.get('char_cv', 0):.3f}\n"
            text_summary += f"Sharpness CV: {text_analysis.get('sharpness_cv', 0):.3f}\n"
            text_summary += f"Noise CV: {text_analysis.get('noise_cv', 0):.3f}\n\n"
            
            if text_analysis.get('outliers'):
                text_summary += f"⚠️ {len(text_analysis['outliers'])} outlier region(s)\n"
            
            if text_analysis.get('is_suspicious'):
                text_summary += "\n🚨 SUSPICIOUS"
            else:
                text_summary += "\n✅ CONSISTENT"
        else:
            text_summary += text_analysis.get('reason', 'N/A')
    else:
        text_summary += text_analysis.get('reason', 'N/A')
    
    ax3.text(0.05, 0.95, text_summary, fontsize=10, family='monospace',
             verticalalignment='top', transform=ax3.transAxes)
    ax3.set_title('Text-to-Text Comparison', fontsize=12, fontweight='bold')
    
    # 4. Boundary analysis
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.axis('off')
    
    boundary_summary = "BOUNDARY ANALYSIS\n\n"
    if 'mean_gradient' in boundary_analysis:
        boundary_summary += f"Mean Gradient: {boundary_analysis['mean_gradient']:.2f}\n"
        boundary_summary += f"Max Gradient: {boundary_analysis['max_gradient']:.2f}\n"
        boundary_summary += f"Sharp Ratio: {boundary_analysis['sharp_ratio']:.2%}\n"
        boundary_summary += f"Boundaries: {boundary_analysis['num_boundaries']}\n\n"
        
        if boundary_analysis.get('is_suspicious'):
            boundary_summary += "🚨 SHARP BOUNDARIES\n(Pasted content)"
        else:
            boundary_summary += "✅ NATURAL BOUNDARIES"
    else:
        boundary_summary += boundary_analysis.get('reason', 'N/A')
    
    ax4.text(0.05, 0.95, boundary_summary, fontsize=10, family='monospace',
             verticalalignment='top', transform=ax4.transAxes)
    ax4.set_title('Boundary Detection', fontsize=12, fontweight='bold')
    
    # 5. Overall verdict
    ax5 = fig.add_subplot(gs[1, 1:])
    ax5.axis('off')
    
    verdict_text = "OVERALL VERDICT\n\n"
    verdict_text += f"Suspicious Signals: {results['suspicious_signals']}/{results['total_checks']}\n"
    verdict_text += f"Confidence: {results['confidence']:.0%}\n\n"
    
    if results['reasons']:
        verdict_text += "Reasons:\n"
        for reason in results['reasons']:
            verdict_text += f"  • {reason}\n"
    
    verdict_text += "\n"
    if results['is_suspicious']:
        verdict_text += "🚨 SUSPICIOUS - Region-aware fraud indicators detected"
    else:
        verdict_text += "✅ CLEAN - No fraud indicators in region-aware analysis"
    
    # Color based on verdict
    bgcolor = '#ffcccc' if results['is_suspicious'] else '#ccffcc'
    ax5.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax5.transAxes,
                                facecolor=bgcolor, alpha=0.3))
    
    ax5.text(0.05, 0.95, verdict_text, fontsize=11, family='monospace',
             verticalalignment='top', transform=ax5.transAxes, fontweight='bold')
    ax5.set_title('Region-Aware Analysis Result', fontsize=14, fontweight='bold')
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Region-aware analysis visualization saved to {output_path}")
