"""
Test Gemini Vision Segmentation

Quick test to verify Gemini can detect regions on a sample document.
"""

import os

from dotenv import load_dotenv

from src.analyzers.region_aware.gemini_segmenter import segment_document_with_gemini, visualize_gemini_segmentation

# Set API key (user needs to set this)
# export GEMINI_API_KEY="your-key-here"
load_dotenv()
test_image = "/Users/vignesh/Downloads/vigneshPassportDevakottai2.png"
output_path = "test_gemini_segmentation.png"

print("Testing Gemini Vision Segmentation...")
print(f"Image: {test_image}")
print(f"API Key set: {bool(os.getenv('GEMINI_API_KEY'))}\n")

result = segment_document_with_gemini(test_image)

if 'error' in result:
    print(f"❌ Error: {result['error']}")
else:
    print(f"✅ Success!")
    print(f"   Regions detected: {result['num_regions']}")
    
    for region in result['regions']:
        print(f"   - {region['type']}: {region.get('description', 'N/A')}")
    
    # Visualize
    visualize_gemini_segmentation(test_image, result, output_path)
    print(f"\nVisualization saved to: {output_path}")
