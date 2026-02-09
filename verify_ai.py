
import cv2
import numpy as np
import os
from src.analyzers import ela, noise, copy_move, metadata, bpcs, frequency, prnu
from src.analyzers.frequence_v2 import FrequencyDeepfakeDetector


def create_synthetic_data(filename="test_ai_fake.jpg"):
    # Create a blank image
    img = np.zeros((500, 500, 3), dtype=np.uint8)
    
    # 1. Simulate "Checkerboard" artifact often found in GANs
    # We add a subtle high-frequency grid
    grid = np.zeros((500, 500), dtype=np.float32)
    grid[::2, ::2] = 1.0 # Checkerboard pattern
    
    # Smooth background
    img[:] = (128, 128, 128)
    
    # Add the grid artifact (very subtle)
    artifact = (grid * 5).astype(np.uint8)
    img[:, :, 0] = cv2.add(img[:, :, 0], artifact)
    
    # Save as JPEG
    cv2.imwrite(filename, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return filename

def verify():
    output_dir = "verification_output_ai"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # image_path = create_synthetic_data()
    # image_path = "/Users/vignesh/Downloads/vigneshPassportDevakottai2.png"
    # image_path = '/Users/vignesh/Documents/Canada Documents/Passport - Front page.jpg'
    # image_path = '/Users/vignesh/Downloads/ArjunvigneshPassport.png'
    image_path = '/Users/vignesh/Downloads/passport fee receipt.jpeg'
    print(f"Created synthetic AI fake: {image_path}")

    print("--- Testing Frequency Analysis ---")
    freq_out = os.path.join(output_dir, "freq_result.png")
    # res = frequency.analyze_frequency(image_path, output_path=freq_out)

    detector = FrequencyDeepfakeDetector()
    res = detector.analyze_image(image_path, output_dir)

    print(f"Energy Ratio: {res}")
    if os.path.exists(freq_out): 
        print("Frequency Analysis: PASSED")
    else:
        print("Frequency Analysis: FAILED")

    print("--- Testing PRNU ---")
    prnu_out = os.path.join(output_dir, "prnu_result.png")
    res = prnu.extract_noise_pattern(image_path, output_path=prnu_out)
    # Synthetic image has no real sensor noise, so variance should be very low or just reflect the grid
    print(f"PRNU Variance: {res}")
    if res.get("error"):
        print(f"PRNU Error: {res.get('error')}")
    
    if os.path.exists(prnu_out):
        print("PRNU Analysis: PASSED")
    else:
        print("PRNU Analysis: FAILED")

if __name__ == "__main__":
    verify()
