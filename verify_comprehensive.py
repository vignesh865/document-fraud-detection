"""
Comprehensive Verification - All Analyzers on Real Documents
"""

import cv2
import numpy as np
from src.analyzers import ela, noise, font_analysis, resolution_analysis
import os

def main():
    """Test all document analyzers on real passports."""
    print("="*80)
    print(" COMPREHENSIVE DOCUMENT FRAUD DETECTION - ALL ANALYZERS")
    print("="*80)
    
    # Create output directory
    output_dir = "verification_output_comprehensive"
    os.makedirs(output_dir, exist_ok=True)
    
    # Test paths
    test_paths = [
        "/Users/vignesh/Downloads/vigneshPassportDevakottai2.png",
        '/Users/vignesh/Documents/Canada Documents/Passport - Front page.jpg',
        '/Users/vignesh/Downloads/ArjunvigneshPassport.png',
        '/Users/vignesh/Downloads/passport fee receipt.jpeg'
    ]
    
    # Process each document
    for idx, test_path in enumerate(test_paths, 1):
        doc_name = os.path.basename(test_path)
        print(f"\n{'='*80}")
        print(f"DOCUMENT {idx}/{len(test_paths)}: {doc_name}")
        print("="*80)
        
        if not os.path.exists(test_path):
            print(f"   ❌ ERROR: File not found: {test_path}")
            continue
        
        # Create subdirectory
        doc_output_dir = os.path.join(output_dir, f"doc{idx}_{os.path.splitext(doc_name)[0]}")
        os.makedirs(doc_output_dir, exist_ok=True)
        
        # Results tracker
        suspicious_flags = []
        
        # 1. Format-Specific Analysis (ELA for JPEG, PNG Analysis for PNG)
        print(f"\n  [1/4] Format-Specific Analysis...")
        print("-"*80)
        
        # Determine format
        from PIL import Image as PILImage
        img_format = PILImage.open(test_path).format
        
        if img_format in ['JPEG', 'JPG']:
            # ELA for JPEG
            ela_output = os.path.join(doc_output_dir, "ela.png")
            format_res = ela.perform_ela(test_path, output_path=ela_output)
            
            if "error" in format_res:
                print(f"   ❌ ERROR: {format_res['error']}")
            else:
                print(f"   ELA - Suspicious Ratio: {format_res['suspicious_ratio']:.2%}")
                print(f"   ELA - Num Regions: {format_res['num_suspicious_regions']}")
                print(f"   Verdict: {'🚨 SUSPICIOUS' if format_res['is_suspicious'] else '✅ CLEAN'}")
                if format_res['is_suspicious']:
                    suspicious_flags.append("ELA")
        
        elif img_format == 'PNG':
            # PNG-specific analysis
            from src.analyzers import png_analysis
            png_output = os.path.join(doc_output_dir, "png.png")
            format_res = png_analysis.analyze_png(test_path, output_path=png_output)
            
            if "error" in format_res:
                print(f"   ❌ ERROR: {format_res['error']}")
            else:
                verdict = format_res['verdict']
                print(f"   PNG Analysis - {verdict['suspicious_signals']}/{verdict['total_checks']} tests flagged")
                print(f"   LSB Suspicious: {format_res['lsb']['is_suspicious']}")
                print(f"   Chunks Suspicious: {format_res['chunks']['is_suspicious']}")
                print(f"   Stats Suspicious: {format_res['stats'].get('is_suspicious', False)}")
                print(f"   Verdict: {'🚨 SUSPICIOUS' if verdict['is_suspicious'] else '✅ CLEAN'}")
                if verdict['is_suspicious']:
                    suspicious_flags.append("PNG")
        
        else:
            print(f"   ⚠️  Unsupported format: {img_format}")
        
        # 2. Noise Variance
        print(f"\n  [2/4] Noise Variance Analysis...")
        print("-"*80)
        noise_output = os.path.join(doc_output_dir, "noise.png")
        noise_res = noise.analyze_noise(test_path, output_path=noise_output)
        
        if "error" in noise_res:
            print(f"   ❌ ERROR: {noise_res['error']}")
        elif 'variance_cv' not in noise_res:
            print(f"   ⚠️  {noise_res.get('interpretation', 'N/A')}")
        else:
            print(f"   Variance CV: {noise_res['variance_cv']:.4f}")
            print(f"   Outlier Ratio: {noise_res['outlier_ratio']:.2%}")
            print(f"   Verdict: {'🚨 SUSPICIOUS' if noise_res['is_suspicious'] else '✅ CLEAN'}")
            if noise_res['is_suspicious']:
                suspicious_flags.append("Noise")
        
        # 3. Font Consistency
        print(f"\n  [3/4] Font Consistency Analysis...")
        print("-"*80)
        font_output = os.path.join(doc_output_dir, "font.png")
        font_res = font_analysis.analyze_font_consistency(test_path, output_path=font_output)
        
        if "error" in font_res:
            print(f"   ❌ ERROR: {font_res['error']}")
        elif 'consistency_score' not in font_res:
            print(f"   ⚠️  {font_res.get('interpretation', 'N/A')}")
        else:
            print(f"   Consistency Score: {font_res['consistency_score']:.4f}")
            print(f"   Num Regions: {font_res['num_regions']}, Outliers: {font_res['num_outliers']}")
            print(f"   Verdict: {'🚨 SUSPICIOUS' if font_res['is_suspicious'] else '✅ CLEAN'}")
            if font_res['is_suspicious']:
                suspicious_flags.append("Font")
        
        # 4. Resolution Inconsistency
        print(f"\n  [4/4] Resolution Inconsistency Detection...")
        print("-"*80)
        res_output = os.path.join(doc_output_dir, "resolution.png")
        res_res = resolution_analysis.analyze_resolution_inconsistency(test_path, output_path=res_output)
        
        if "error" in res_res:
            print(f"   ❌ ERROR: {res_res['error']}")
        elif 'cv_sharpness' not in res_res:
            print(f"   ⚠️  {res_res.get('interpretation', 'N/A')}")
        else:
            print(f"   CV Sharpness: {res_res['cv_sharpness']:.4f}")
            print(f"   Outlier Ratio: {res_res['outlier_ratio']:.2%}")
            print(f"   Verdict: {'🚨 SUSPICIOUS' if res_res['is_suspicious'] else '✅ CLEAN'}")
            if res_res['is_suspicious']:
                suspicious_flags.append("Resolution")
        
        # OVERALL VERDICT
        print(f"\n  ╔{'═'*76}╗")
        print(f"  ║ OVERALL VERDICT{' '*60}║")
        print(f"  ╚{'═'*76}╝")
        
        if len(suspicious_flags) == 0:
            print(f"   ✅ CLEAN - No fraud indicators detected")
        elif len(suspicious_flags) <= 2:
            print(f"   ⚠️  SUSPICIOUS - {len(suspicious_flags)}/4 tests flagged")
            print(f"      Flagged by: {', '.join(suspicious_flags)}")
        else:
            print(f"   🚨 HIGHLY SUSPICIOUS - {len(suspicious_flags)}/4 tests flagged")
            print(f"      Flagged by: {', '.join(suspicious_flags)}")
            print(f"      ⚠️  STRONG indication of fraud/manipulation")
    
    print("\n" + "="*80)
    print("VERIFICATION COMPLETE")
    print("="*80)
    print(f"\nResults: {output_dir}/")
    print("Check subdirectories for visualizations and analysis.")

if __name__ == "__main__":
    main()
