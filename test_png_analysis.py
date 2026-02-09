"""
Test PNG Analysis on Real vs Fake PNGs

Now with a REAL PNG to properly test false positive rate.
"""

import os
from src.analyzers.png_analysis import analyze_png

def main():
    print("="*80)
    print(" PNG FRAUD DETECTION TEST - REAL vs FAKE")
    print("="*80)
    
    output_dir = "verification_output_png_test"
    os.makedirs(output_dir, exist_ok=True)
    
    # Test documents
    documents = [
        {
            'path': "/Users/vignesh/Downloads/vigneshPassportDevakottai2.png",
            'name': "vigneshPassportDevakottai2.png",
            'ground_truth': "FAKE"
        },
        {
            'path': '/Users/vignesh/Downloads/ArjunvigneshPassport.png',
            'name': "ArjunvigneshPassport.png",
            'ground_truth': "FAKE"
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
        print(f"[{idx}/3] {doc['name']}")
        print(f"Ground Truth: {doc['ground_truth']}")
        print("="*80)
        
        if not os.path.exists(doc['path']):
            print(f"❌ File not found: {doc['path']}")
            continue
        
        # Create output path
        doc_dir = os.path.join(output_dir, f"doc{idx}_{os.path.splitext(doc['name'])[0]}")
        os.makedirs(doc_dir, exist_ok=True)
        output_path = os.path.join(doc_dir, "png_analysis.png")
        
        # Run PNG analysis
        result = analyze_png(doc['path'], output_path=output_path)
        
        # Display results
        print(f"\n📊 PNG ANALYSIS:")
        
        # LSB
        lsb = result.get('lsb', {})
        print(f"\n   LSB Analysis:")
        print(f"      Randomness: {lsb.get('randomness_score', 0):.3f}")
        print(f"      Chi-square: {lsb.get('chi_square', 0):.2f}")
        print(f"      Anomalies: {lsb.get('num_anomalies', 0)}")
        print(f"      Suspicious: {'🚨 YES' if lsb.get('is_suspicious') else '✅ NO'}")
        
        # Chunks
        chunks = result.get('chunks', {})
        print(f"\n   PNG Chunk Analysis:")
        print(f"      Editor found: {chunks.get('has_editor_metadata', False)}")
        print(f"      Suspicious chunks: {len(chunks.get('suspicious_chunks', []))}")
        print(f"      Suspicious: {'🚨 YES' if chunks.get('is_suspicious') else '✅ NO'}")
        
        # Stats
        stats = result.get('stats', {})
        print(f"\n   Statistical Analysis:")
        print(f"      Outlier regions: {stats.get('num_outlier_regions', 0)}")
        print(f"      Suspicious: {'🚨 YES' if stats.get('is_suspicious') else '✅ NO'}")
        
        # Overall
        is_suspicious = result.get('is_suspicious', False)
        suspicious_count = sum([
            lsb.get('is_suspicious', False),
            chunks.get('is_suspicious', False),
            stats.get('is_suspicious', False)
        ])
        
        print(f"\n   Overall: {suspicious_count}/3 tests flagged")
        
        # Verdict
        prediction = "FAKE" if is_suspicious else "REAL"
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
            'flags': suspicious_count
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
        print(f"{status} {r['name']:<45} GT: {r['ground_truth']:<4} → Pred: {r['prediction']:<4} ({r['flags']}/3 flags)")
    
    print(f"\n{'='*80}")
    print(f"Visualizations saved to: {output_dir}/")
    print("="*80)

if __name__ == "__main__":
    main()
