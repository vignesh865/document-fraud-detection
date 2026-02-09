"""
Test Region-Aware Fraud Detection on Ground Truth

Ground truth from user:
- REAL: Passport - Front page.jpg, passport fee receipt.jpeg
- FAKE: vigneshPassportDevakottai2.png, ArjunvigneshPassport.png
"""

import os

from dotenv import load_dotenv

from src.analyzers.region_aware import analyze_document
load_dotenv()
def main():
    print("="*80)
    print(" REGION-AWARE FRAUD DETECTION TEST")
    print("="*80)
    
    output_dir = "verification_output_region_aware"
    os.makedirs(output_dir, exist_ok=True)
    
    # Test documents with ground truth
    documents = [
        {
            'path': "/Users/vignesh/Downloads/vigneshPassportDevakottai2.png",
            'name': "vigneshPassportDevakottai2.png",
            'ground_truth': "FAKE"
        },
        {
            'path': '/Users/vignesh/Documents/Canada Documents/Passport - Front page.jpg',
            'name': "Passport - Front page.jpg",
            'ground_truth': "REAL"
        },
        {
            'path': '/Users/vignesh/Downloads/ArjunvigneshPassport.png',
            'name': "ArjunvigneshPassport.png",
            'ground_truth': "FAKE"
        },
        {
            'path': '/Users/vignesh/Downloads/passport fee receipt.jpeg',
            'name': "passport fee receipt.jpeg",
            'ground_truth': "REAL"
        }
    ]
    
    results_summary = []
    
    for idx, doc in enumerate(documents, 1):
        print(f"\n{'='*80}")
        print(f"[{idx}/4] {doc['name']}")
        print(f"Ground Truth: {doc['ground_truth']}")
        print("="*80)
        
        if not os.path.exists(doc['path']):
            print(f"❌ File not found: {doc['path']}")
            continue
        
        # Create output path
        doc_dir = os.path.join(output_dir, f"doc{idx}_{os.path.splitext(doc['name'])[0]}")
        os.makedirs(doc_dir, exist_ok=True)
        output_path = os.path.join(doc_dir, "region_aware_analysis.png")
        
        # Run region-aware analysis
        result = analyze_document(doc['path'], output_path=output_path)
        
        if 'error' in result:
            print(f"❌ ERROR: {result['error']}")
            continue
        
        # Display results
        print(f"\n📊 SEGMENTATION:")
        print(f"   Total regions: {result['segmentation']['num_regions']}")
        print(f"   Text regions: {result['segmentation']['num_text_regions']}")
        print(f"   Photo regions: {result['segmentation']['num_photo_regions']}")
        
        print(f"\n📝 TEXT ANALYSIS (text-to-text only):")
        text_res = result['text_analysis']
        if 'stroke_cv' in text_res:
            print(f"   Stroke CV: {text_res['stroke_cv']:.3f}")
            print(f"   Char CV: {text_res['char_cv']:.3f}")
            print(f"   Sharpness CV: {text_res['sharpness_cv']:.3f}")
            print(f"   Verdict: {'🚨 SUSPICIOUS' if text_res['is_suspicious'] else '✅ CONSISTENT'}")
        else:
            print(f"   {text_res.get('reason', 'N/A')}")
        
        print(f"\n🔲 BOUNDARY ANALYSIS:")
        bound_res = result['boundary_analysis']
        if 'sharp_ratio' in bound_res:
            print(f"   Sharp Ratio: {bound_res['sharp_ratio']:.2%}")
            print(f"   Max Gradient: {bound_res['max_gradient']:.1f}")
            print(f"   Verdict: {'🚨 SHARP BOUNDARIES' if bound_res['is_suspicious'] else '✅ NATURAL'}")
        else:
            print(f"   {bound_res.get('reason', 'N/A')}")
        
        # Overall verdict
        prediction = "FAKE" if result['is_suspicious'] else "REAL"
        correct = (prediction == doc['ground_truth'])
        
        print(f"\n╔{'═'*76}╗")
        print(f"║ VERDICT: {prediction:^67} ║")
        print(f"║ Ground Truth: {doc['ground_truth']:^60} ║")
        print(f"║ Result: {'✅ CORRECT' if correct else '❌ INCORRECT':^66} ║")
        print(f"╚{'═'*76}╝")
        
        results_summary.append({
            'name': doc['name'],
            'ground_truth': doc['ground_truth'],
            'prediction': prediction,
            'correct': correct,
            'confidence': result['confidence']
        })
    
    # Summary
    print(f"\n\n{'='*80}")
    print(" FINAL RESULTS")
    print("="*80)
    
    correct_count = sum(1 for r in results_summary if r['correct'])
    total = len(results_summary)
    accuracy = correct_count / total if total > 0 else 0
    
    print(f"\nAccuracy: {correct_count}/{total} = {accuracy:.0%}\n")
    
    for r in results_summary:
        status = "✅" if r['correct'] else "❌"
        print(f"{status} {r['name']:<40} GT: {r['ground_truth']:<4} → Pred: {r['prediction']:<4} (Conf: {r['confidence']:.0%})")
    
    print(f"\n{'='*80}")
    print(f"Visualizations saved to: {output_dir}/")
    print("="*80)

if __name__ == "__main__":
    main()
