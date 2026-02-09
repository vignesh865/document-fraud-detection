"""
Verification script for document-specific fraud detection
Tests Font Consistency and Resolution Inconsistency analyzers
"""

import cv2
import numpy as np
from src.analyzers import font_analysis, resolution_analysis
import os

def main():
    """Test document-specific analyzers on multiple documents."""
    print("="*70)
    print("DOCUMENT-SPECIFIC FRAUD DETECTION - VERIFICATION")
    print("="*70)
    
    # Create output directory
    output_dir = "verification_output_document"
    os.makedirs(output_dir, exist_ok=True)
    
    # Test paths provided by user
    test_paths = [
        "/Users/vignesh/Downloads/vigneshPassportDevakottai2.png",
        '/Users/vignesh/Documents/Canada Documents/Passport - Front page.jpg',
        '/Users/vignesh/Downloads/ArjunvigneshPassport.png',
        '/Users/vignesh/Downloads/passport fee receipt.jpeg'
    ]
    
    # Process each document
    for idx, test_path in enumerate(test_paths, 1):
        doc_name = os.path.basename(test_path)
        print(f"\n{'='*70}")
        print(f"DOCUMENT {idx}/{len(test_paths)}: {doc_name}")
        print("="*70)
        
        if not os.path.exists(test_path):
            print(f"   ❌ ERROR: File not found: {test_path}")
            continue
        
        # Create subdirectory for this document
        doc_output_dir = os.path.join(output_dir, f"doc{idx}_{os.path.splitext(doc_name)[0]}")
        os.makedirs(doc_output_dir, exist_ok=True)
        
        # Test Font Consistency Analysis
        print(f"\n  Testing Font Consistency Analysis...")
        print("-"*70)
        font_output = os.path.join(doc_output_dir, "font_analysis.png")
        font_results = font_analysis.analyze_font_consistency(test_path, output_path=font_output)
        
        if "error" in font_results:
            print(f"   ❌ ERROR: {font_results['error']}")
        elif 'consistency_score' not in font_results:
            print(f"   ⚠️  Insufficient text regions detected")
            print(f"   {font_results.get('interpretation', 'N/A')}")
        else:
            print(f"   Consistency Score: {font_results['consistency_score']:.4f}")
            print(f"   Stroke Width CV: {font_results['stroke_width_cv']:.4f}")
            print(f"   Char Height CV: {font_results['char_height_cv']:.4f}")
            print(f"   Num Regions: {font_results['num_regions']}")
            print(f"   Num Outliers: {font_results['num_outliers']}")
            print(f"   Suspicious: {'🚨 YES' if font_results['is_suspicious'] else '✅ NO'}")
        
        # Test Resolution Inconsistency Detection
        print(f"\n  Testing Resolution Inconsistency Detection...")
        print("-"*70)
        res_output = os.path.join(doc_output_dir, "resolution_analysis.png")
        res_results = resolution_analysis.analyze_resolution_inconsistency(test_path, output_path=res_output)
        
        if "error" in res_results:
            print(f"   ❌ ERROR: {res_results['error']}")
        elif 'cv_sharpness' not in res_results:
            print(f"   ⚠️  Insufficient patches for analysis")
            print(f"   {res_results.get('interpretation', 'N/A')}")
        else:
            print(f"   CV Sharpness: {res_results['cv_sharpness']:.4f}")
            print(f"   Outlier Ratio: {res_results['outlier_ratio']:.2%}")
            print(f"   Num Outliers: {res_results['num_outliers']}/{res_results['total_patches']}")
            print(f"   Mean Sharpness: {res_results['mean_sharpness']:.2f}")
            print(f"   Suspicious: {'🚨 YES' if res_results['is_suspicious'] else '✅ NO'}")
        
        # Overall verdict
        print(f"\n  OVERALL VERDICT:")
        print("-"*70)
        font_sus = font_results.get('is_suspicious', False) if 'error' not in font_results else False
        res_sus = res_results.get('is_suspicious', False) if 'error' not in res_results else False
        
        if font_sus or res_sus:
            print(f"   🚨 SUSPICIOUS - Potential fraud detected")
            if font_sus:
                print(f"      - Font inconsistency detected")
            if res_sus:
                print(f"      - Resolution inconsistency detected")
        else:
            print(f"   ✅ CLEAN - No document fraud indicators")
    
    print("\n" + "="*70)
    print("VERIFICATION COMPLETE")
    print("="*70)
    print(f"\nResults saved to: {output_dir}/")
    print("Check subdirectories for individual document analysis visualizations.")

if __name__ == "__main__":
    main()
