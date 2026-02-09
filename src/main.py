
import typer
import os
from src.analyzers import ela, noise, copy_move, metadata, bpcs, mrz_checker, pdf_analysis, frequency, prnu

app = typer.Typer()

@app.command()
def analyze(image_path: str, output_dir: str = "output", mrz: str = None):
    """
    Run fraud detection analysis on a file (Image or PDF).
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Analyzing {image_path}...")
    
    filename = os.path.basename(image_path)
    name, ext = os.path.splitext(filename)
    ext = ext.lower()

    # Document Type Detection
    is_pdf = ext == '.pdf'
    is_image = ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']

    if is_image:
        # 1. Error Level Analysis
        print("Running ELA...")
        ela_output = os.path.join(output_dir, f"{name}_ela.jpg")
        ela.perform_ela(image_path, output_path=ela_output)

        # 2. Noise Analysis
        print("Running Noise Analysis...")
        noise_output = os.path.join(output_dir, f"{name}_noise.jpg")
        noise.analyze_noise(image_path, output_path=noise_output)
        
        # 3. Copy-Move Detection
        print("Running Copy-Move Detection...")
        copy_move_output = os.path.join(output_dir, f"{name}_copy_move.jpg")
        copy_move.detect_copy_move(image_path, output_path=copy_move_output)

        # 4. Metadata Analysis
        print("Running Metadata Analysis...")
        meta_output = os.path.join(output_dir, f"{name}_metadata.txt")
        metadata.analyze_metadata(image_path, output_path=meta_output)

        # 5. BPCS
        print("Running BPCS Analysis...")
        bpcs_output = os.path.join(output_dir, f"{name}_bpcs.png")
        bpcs.analyze_bpcs(image_path, output_path=bpcs_output)

        # 6. Frequency Analysis (Deepfake/GAN)
        print("Running Frequency Analysis...")
        freq_output = os.path.join(output_dir, f"{name}_freq.png")
        freq_res = frequency.analyze_frequency(image_path, output_path=freq_output)
        freq_ratio = freq_res.get('energy_ratio', 0)
        freq_suspicious = freq_res.get('is_suspicious', False)
        print(f"Frequency Ratio: {freq_ratio:.4f} [{'SUSPICIOUS' if freq_suspicious else 'OK'}]")

        # 7. PRNU Analysis
        print("Running PRNU Analysis...")
        prnu_output = os.path.join(output_dir, f"{name}_prnu.png")
        prnu_res = prnu.extract_noise_pattern(image_path, output_path=prnu_output)
        prnu_var = prnu_res.get('noise_variance', 0)
        prnu_suspicious = prnu_res.get('is_suspicious', False)
        print(f"PRNU Variance: {prnu_var:.6e} [{'SUSPICIOUS' if prnu_suspicious else 'OK'}]")

    elif is_pdf:
        print("PDF detected. Running Structure Analysis...")
        pdf_out = os.path.join(output_dir, f"{name}_pdf_analysis.txt")
        pdf_analysis.analyze_pdf_structure(image_path, output_path=pdf_out)
    
    else:
        print(f"Unsupported file format: {ext}")

    # MRZ Check (if provided)
    if mrz:
        print("Validating MRZ...")
        result = mrz_checker.validate_mrz(mrz)
        print(f"MRZ Valid: {result['valid']}")
        print(f"Details: {result['details']}")

    print(f"Analysis complete. Results saved to {output_dir}")

@app.command()
def check_mrz(mrz_text: str):
    """
    Validate an MRZ string directly.
    """
    result = mrz_checker.validate_mrz(mrz_text)
    print(f"MRZ Valid: {result['valid']}")
    print(f"Type: {result['type']}")
    for detail in result['details']:
        print(f"- {detail}")

if __name__ == "__main__":
    app()
