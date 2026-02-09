
import os
import exifread

def analyze_metadata(image_path, output_path=None):
    """
    Analyzes image metadata (EXIF) for signs of editing using exifread.
    Checks for:
    - Software tags (e.g., Photoshop, GIMP)
    - Discrepancies between Original and Digitized timestamps
    
    Args:
        image_path (str): Path to the image file.
        output_path (str, optional): Path to save a text report.
        
    Returns:
        dict: A dictionary containing suspicious flags and raw metadata.
    """
    results = {
        "suspicious_tags": [],
        "software": "Unknown",
        "dates": {}
    }

    try:
        with open(image_path, 'rb') as f:
            tags = exifread.process_file(f)
            
            # Check Software Tag
            if 'Image Software' in tags:
                software = str(tags['Image Software'])
                results["software"] = software
                
                suspicious_keywords = ["photoshop", "gimp", "paint", "adobe", "editor"]
                if any(k in software.lower() for k in suspicious_keywords):
                    results["suspicious_tags"].append(f"Editing software detected: {software}")

            # Check Dates
            if 'Image DateTime' in tags:
                results["dates"]["modify_date"] = str(tags['Image DateTime'])
                
            if 'EXIF DateTimeOriginal' in tags:
                results["dates"]["original_date"] = str(tags['EXIF DateTimeOriginal'])
                
            if 'EXIF DateTimeDigitized' in tags:
                results["dates"]["digitized_date"] = str(tags['EXIF DateTimeDigitized'])

    except Exception as e:
        results["error"] = str(e)
        
    if output_path:
        with open(output_path, "w") as f:
            for key, value in results.items():
                f.write(f"{key}: {value}\n")
        print(f"Metadata report saved to {output_path}")

    return results
