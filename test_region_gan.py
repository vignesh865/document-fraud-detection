"""
Test Region-Based GAN Detection

Tests GAN detection on individual regions from Gemini segmentation.
"""

import os
from dotenv import load_dotenv
from src.analyzers.region_gan_detector import analyze_document_regions_with_gan, visualize_gan_regions

# Load environment variables
load_dotenv()

def main():
    print("="*80)
    print(" REGION-BASED GAN DETECTION TEST")
    print("="*80)
    
    output_dir = "verification_output_region_gan"
    os.makedirs(output_dir, exist_ok=True)
    
    # Test documents
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
        },
        {
            'path': '/Users/vignesh/Downloads/2passport fee receipt.png',
            'name': "2passport fee receipt.png",
            'ground_truth': "REAL"
        }
    ]
    
    results_summary = []
    
    for idx, doc in enumerate(documents, 1):
        print(f"\n{'='*80}")
        print(f"[{idx}/5] {doc['name']}")
        print(f"Ground Truth: {doc['ground_truth']}")
        print("="*80)
        
        if not os.path.exists(doc['path']):
            print(f"❌ File not found: {doc['path']}")
            continue
        
        # Create output path
        doc_dir = os.path.join(output_dir, f"doc{idx}_{os.path.splitext(doc['name'])[0]}")
        os.makedirs(doc_dir, exist_ok=True)
        output_path = os.path.join(doc_dir, "gan_region_analysis.png")
        
        # Run region-based GAN detection
        result = analyze_document_regions_with_gan(doc['path'])
        
        if 'error' in result:
            print(f"❌ ERROR: {result['error']}")
            continue
        
        # Display results
        print(f"\n📊 REGION ANALYSIS:")
        print(f"   Total regions detected: {result['total_regions']}")
        print(f"   Regions analyzed: {result['analyzed_regions']}")
        print(f"   Suspicious regions: {result['suspicious_regions_count']}")
        print(f"   Suspicion ratio: {result['suspicion_ratio']:.1%}")
        
        if result.get('suspicious_regions'):
            print(f"\n   🚨 SUSPICIOUS REGIONS:")
            for sr in result['suspicious_regions']:
                print(f"      • {sr['type']}: {sr['description']} (GAN: {sr['confidence']:.0%})")
        
        # Verdict
        prediction = "FAKE" if result['is_suspicious'] else "REAL"
        correct = (prediction == doc['ground_truth'])
        
        print(f"\n╔{'═'*76}╗")
        print(f"║ VERDICT: {prediction:^67} ║")
        print(f"║ Ground Truth: {doc['ground_truth']:^60} ║")
        print(f"║ Result: {'✅ CORRECT' if correct else '❌ INCORRECT':^66} ║")
        print(f"╚{'═'*76}╝")
        
        # Visualize
        visualize_gan_regions(doc['path'], result, output_path)
        
        results_summary.append({
            'name': doc['name'],
            'ground_truth': doc['ground_truth'],
            'prediction': prediction,
            'correct': correct,
            'suspicion_ratio': result['suspicion_ratio']
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
        print(f"{status} {r['name']:<45} GT: {r['ground_truth']:<4} → Pred: {r['prediction']:<4} ({r['suspicion_ratio']:.0%} suspicious)")
    
    print(f"\n{'='*80}")
    print(f"Visualizations saved to: {output_dir}/")
    print("="*80)

if __name__ == "__main__":
    main()
