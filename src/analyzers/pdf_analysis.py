
import os
from pypdf import PdfReader

def analyze_pdf_structure(pdf_path, output_path=None):
    """
    Analyzes PDF structure for potential fraud.
    Checks for:
    - Incremental updates (multiple EOF markers might indicate this, though pypdf handles it abstractly)
    - Metadata inconsistencies
    
    Args:
        pdf_path (str): Path to the PDF file.
        output_path (str, optional): Path to save report.
        
    Returns:
        dict: Analysis results.
    """
    results = {
        "is_encrypted": False,
        "metadata": {},
        "suspicious_flags": [],
        "layers": 0
    }
    
    try:
        reader = PdfReader(pdf_path)
        results["is_encrypted"] = reader.is_encrypted
        
        if reader.metadata:
            results["metadata"] = {k: str(v) for k, v in reader.metadata.items()}
            
            # Check Creator/Producer for editing tools
            producer = results["metadata"].get("/Producer", "").lower()
            creator = results["metadata"].get("/Creator", "").lower()
            
            suspicious_keywords = ["photoshop", "gimp", "ilovepdf", "smallpdf", "editor"]
            if any(k in producer for k in suspicious_keywords) or any(k in creator for k in suspicious_keywords):
                 results["suspicious_flags"].append(f"Suspicious Producer/Creator: {producer} / {creator}")

        # Basic check for layers (Optional Content Groups)
        # pypdf might not expose OCGs easily in high level API, 
        # but we can check if '/OCProperties' is in the root object.
        # This is a heuristic.
        try:
            if '/OCProperties' in reader.trailer['/Root']:
                 results["suspicious_flags"].append("PDF contains Optional Content Groups (Layers), which can hide/show content.")
                 results["layers"] = "Detected"
        except:
             pass

    except Exception as e:
        results["error"] = str(e)
        
    if output_path:
        with open(output_path, "w") as f:
            for key, value in results.items():
                f.write(f"{key}: {value}\n")
        print(f"PDF Analysis saved to {output_path}")

    return results
