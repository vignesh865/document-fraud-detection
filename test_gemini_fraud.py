"""
Test Gemini AI Direct Fraud Detection

Tests all 4 documents with Gemini's direct fraud analysis.
"""

import os
from dotenv import load_dotenv
from src.analyzers.gemini_fraud_detector import analyze_with_gemini, visualize_analysis

# Load environment variables
load_dotenv()

def main():
    print("="*80)
    print(" GEMINI AI DIRECT FRAUD DETECTION TEST")
    print("="*80)
    
    output_dir = "verification_output_gemini_ai"
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
        output_path = os.path.join(doc_dir, "gemini_ai_analysis.png")
        
        # Run Gemini AI analysis
        result = analyze_with_gemini(doc['path'])
        
        if 'error' in result:
            print(f"❌ ERROR: {result['error']}")
            if 'raw_response' in result:
                print(f"Raw response: {result['raw_response'][:200]}...")
            continue
        
        # Display results
        print(f"\n🤖 GEMINI AI ANALYSIS:")
        print(f"   Verdict: {'🚨 FORGED' if result['is_forged'] else '✅ AUTHENTIC'}")
        print(f"   Confidence: {result['confidence']*100:.0f}%")
        
        if result.get('red_flags'):
            print(f"\n   🚨 RED FLAGS:")
            for flag in result['red_flags']:
                print(f"      • {flag}")
        
        if result.get('green_flags'):
            print(f"\n   ✅ GREEN FLAGS:")
            for flag in result['green_flags']:
                print(f"      • {flag}")
        
        if result.get('explanation'):
            print(f"\n   📝 {result['explanation']}")
        
        # Overall verdict
        prediction = "FAKE" if result['is_forged'] else "REAL"
        correct = (prediction == doc['ground_truth'])
        
        print(f"\n╔{'═'*76}╗")
        print(f"║ VERDICT: {prediction:^67} ║")
        print(f"║ Ground Truth: {doc['ground_truth']:^60} ║")
        print(f"║ Result: {'✅ CORRECT' if correct else '❌ INCORRECT':^66} ║")
        print(f"╚{'═'*76}╝")
        
        # Visualize
        visualize_analysis(doc['path'], result, output_path)
        
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
