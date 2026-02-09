"""
PNG-Specific Fraud Detection

Techniques that work on lossless PNG images:
1. LSB (Least Significant Bit) Analysis - detect hidden data/steganography
2. PNG Chunk Analysis - check metadata for manipulation signs
3. Statistical Pixel Analysis - detect splicing via histogram anomalies
"""

import cv2
import numpy as np
from PIL import Image
import struct
from typing import Dict, List
import zlib

def analyze_png(image_path: str, output_path: str = None) -> Dict:
    """
    Comprehensive PNG-specific fraud detection.
    
    Args:
        image_path: Path to PNG image
        output_path: Optional path to save visualization
        
    Returns:
        Dictionary with analysis results
    """
    try:
        # Verify it's actually a PNG
        img_pil = Image.open(image_path)
        if img_pil.format != 'PNG':
            return {
                "is_suspicious": False,
                "skipped": True,
                "reason": f"This analyzer only works on PNG (got {img_pil.format})"
            }
        
        results = {}
        
        # 1. LSB Analysis
        results['lsb'] = _analyze_lsb(image_path)
        
        # 2. PNG Chunk Analysis
        results['chunks'] = _analyze_png_chunks(image_path)
        
        # 3. Statistical Analysis
        results['stats'] = _analyze_pixel_statistics(image_path)
        
        # Overall verdict
        suspicious_signals = 0
        if results['lsb'].get('is_suspicious', False):
            suspicious_signals += 1
        if results['chunks'].get('is_suspicious', False):
            suspicious_signals += 1
        if results['stats'].get('is_suspicious', False):
            suspicious_signals += 1
        
        is_suspicious = suspicious_signals >= 2
        
        results['verdict'] = {
            'is_suspicious': is_suspicious,
            'suspicious_signals': suspicious_signals,
            'total_checks': 3,
            'interpretation': f"{suspicious_signals}/3 PNG-specific tests flagged"
        }
        
        # Visualization
        if output_path:
            _visualize_png_analysis(image_path, results, output_path)
        
        return results
        
    except Exception as e:
        return {"error": str(e)}


def _analyze_lsb(image_path: str) -> Dict:
    """
    Analyze Least Significant Bit plane for hidden data or tampering.
    
    Natural images have random LSB. Hidden data or pasted regions
    have non-random LSB patterns.
    """
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    
    # Extract LSB plane
    lsb_plane = img & 1  # Get least significant bit
    
    # Convert to 0-255 for analysis
    lsb_visual = lsb_plane * 255
    
    # Analyze randomness
    # Natural LSB should be ~50% 0s and 50% 1s
    lsb_mean = np.mean(lsb_plane)
    
    # Chi-square test for randomness
    # Count transitions (01 or 10 patterns indicate randomness)
    transitions = np.sum(np.abs(np.diff(lsb_plane.flatten())))
    max_transitions = lsb_plane.size - 1
    transition_ratio = transitions / max_transitions
    
    # Natural images: ~0.5 transitions
    # Hidden data: usually lower or structured patterns
    is_suspicious = (transition_ratio < 0.4) or (transition_ratio > 0.6) or \
                    (abs(lsb_mean - 0.5) > 0.15)
    
    return {
        'lsb_mean': float(lsb_mean),
        'transition_ratio': float(transition_ratio),
        'is_suspicious': bool(is_suspicious),
        'interpretation': 'LSB should be random (~0.5). Deviation suggests hidden data or editing.'
    }


def _analyze_png_chunks(image_path: str) -> Dict:
    """
    Analyze PNG chunk structure for signs of manipulation.
    
    PNG files are composed of chunks. Suspicious signs:
    - Missing standard chunks
    - Extra tEXt/iTXt chunks (metadata added by editors)
    - Multiple IDAT chunks (re-saved/edited)
    - Suspicious software in metadata
    """
    with open(image_path, 'rb') as f:
        # Read PNG signature
        signature = f.read(8)
        if signature != b'\x89PNG\r\n\x1a\n':
            return {"error": "Not a valid PNG file"}
        
        chunks = []
        suspicious_flags = []
        
        while True:
            # Read chunk
            length_bytes = f.read(4)
            if len(length_bytes) < 4:
                break
            
            length = struct.unpack('>I', length_bytes)[0]
            chunk_type = f.read(4).decode('ascii', errors='ignore')
            chunk_data = f.read(length)
            crc = f.read(4)
            
            chunks.append({
                'type': chunk_type,
                'length': length
            })
            
            # Check for suspicious chunks
            if chunk_type in ['tEXt', 'iTXt', 'zTXt']:
                # Text chunks often added by editing software
                try:
                    text_content = chunk_data.decode('latin1', errors='ignore')
                    if any(keyword in text_content.lower() for keyword in 
                          ['photoshop', 'gimp', 'paint', 'edit', 'adobe']):
                        suspicious_flags.append(f"Editor metadata: {chunk_type}")
                except:
                    pass
        
        # Analyze chunk patterns
        chunk_types = [c['type'] for c in chunks]
        idat_count = chunk_types.count('IDAT')
        
        # Multiple IDAT chunks can be normal, but very many suggests re-encoding
        if idat_count > 20:
            suspicious_flags.append(f"Many IDAT chunks ({idat_count})")
        
        # Check for standard chunks
        if 'IHDR' not in chunk_types:
            suspicious_flags.append("Missing IHDR header")
        
        is_suspicious = len(suspicious_flags) > 0
        
        return {
            'num_chunks': len(chunks),
            'idat_count': idat_count,
            'suspicious_flags': suspicious_flags,
            'is_suspicious': bool(is_suspicious),
            'interpretation': 'Editor metadata or unusual chunk structure suggests manipulation'
        }


def _analyze_pixel_statistics(image_path: str) -> Dict:
    """
    Analyze pixel distribution for splicing detection.
    
    Spliced regions often have different statistical properties
    (histogram shape, color distribution) than the rest of the image.
    """
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Divide into regions and analyze histograms
    patch_size = 128
    h, w = gray.shape
    rows = h // patch_size
    cols = w // patch_size
    
    if rows < 2 or cols < 2:
        return {
            'is_suspicious': False,
            'interpretation': 'Image too small for regional analysis'
        }
    
    # Compute histogram for each patch
    patch_entropies = []
    patch_means = []
    patch_stds = []
    
    for i in range(rows):
        for j in range(cols):
            y1, y2 = i * patch_size, (i + 1) * patch_size
            x1, x2 = j * patch_size, (j + 1) * patch_size
            
            patch = gray[y1:y2, x1:x2]
            
            # Compute entropy (measure of randomness)
            hist, _ = np.histogram(patch, bins=256, range=(0, 256))
            hist = hist / hist.sum()
            hist = hist[hist > 0]
            entropy = -np.sum(hist * np.log2(hist))
            
            patch_entropies.append(entropy)
            patch_means.append(np.mean(patch))
            patch_stds.append(np.std(patch))
    
    patch_entropies = np.array(patch_entropies)
    patch_means = np.array(patch_means)
    patch_stds = np.array(patch_stds)
    
    # Compute variance across patches
    entropy_cv = np.std(patch_entropies) / (np.mean(patch_entropies) + 1e-5)
    mean_cv = np.std(patch_means) / (np.mean(patch_means) + 1e-5)
    
    # High variation suggests spliced regions
    is_suspicious = (entropy_cv > 0.15) or (mean_cv > 0.3)
    
    return {
        'entropy_cv': float(entropy_cv),
        'mean_cv': float(mean_cv),
        'num_patches': len(patch_entropies),
        'is_suspicious': bool(is_suspicious),
        'interpretation': 'High variation across regions suggests spliced content'
    }


def _visualize_png_analysis(image_path: str, results: Dict, output_path: str):
    """Generate visualization of PNG analysis."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Extract LSB plane
    lsb_plane = (img[:, :, 0] & 1) * 255  # Use first channel
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Original
    axes[0, 0].imshow(img_rgb)
    axes[0, 0].set_title('Original PNG', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    # LSB plane
    axes[0, 1].imshow(lsb_plane, cmap='gray')
    lsb_res = results['lsb']
    axes[0, 1].set_title(f'LSB Plane\nMean: {lsb_res["lsb_mean"]:.3f}, '
                         f'Transitions: {lsb_res["transition_ratio"]:.3f}',
                         fontsize=11, fontweight='bold')
    axes[0, 1].axis('off')
    
    # Chunk analysis text
    axes[1, 0].axis('off')
    chunk_res = results['chunks']
    chunk_text = f"PNG Chunk Analysis\n\n"
    chunk_text += f"Total Chunks: {chunk_res['num_chunks']}\n"
    chunk_text += f"IDAT Chunks: {chunk_res['idat_count']}\n\n"
    if chunk_res['suspicious_flags']:
        chunk_text += "⚠️ Suspicious Flags:\n"
        for flag in chunk_res['suspicious_flags']:
            chunk_text += f"  • {flag}\n"
    else:
        chunk_text += "✅ No suspicious flags"
    
    axes[1, 0].text(0.1, 0.5, chunk_text, fontsize=10, family='monospace',
                    verticalalignment='center')
    axes[1, 0].set_title('Chunk Analysis', fontsize=12, fontweight='bold')
    
    # Stats analysis
    axes[1, 1].axis('off')
    stats_res = results['stats']
    stats_text = f"Statistical Analysis\n\n"
    if 'entropy_cv' in stats_res:
        stats_text += f"Entropy CV: {stats_res['entropy_cv']:.4f}\n"
        stats_text += f"Mean CV: {stats_res['mean_cv']:.4f}\n"
        stats_text += f"Patches: {stats_res['num_patches']}\n\n"
        stats_text += "✅ Normal" if not stats_res['is_suspicious'] else "⚠️ Suspicious"
    else:
        stats_text += stats_res.get('interpretation', 'N/A')
    
    axes[1, 1].text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
                    verticalalignment='center')
    axes[1, 1].set_title('Statistical Analysis', fontsize=12, fontweight='bold')
    
    # Overall verdict
    verdict = results['verdict']
    fig.suptitle(f"PNG Analysis - {'🚨 SUSPICIOUS' if verdict['is_suspicious'] else '✅ CLEAN'} "
                 f"({verdict['suspicious_signals']}/{verdict['total_checks']} tests flagged)",
                 fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"PNG analysis visualization saved to {output_path}")
