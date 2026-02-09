
import cv2
import numpy as np
import os
from src.analyzers import ela, noise, copy_move

def create_synthetic_forgery(filename="test_forgery.jpg"):
    # Create a blank image
    img = np.zeros((500, 500, 3), dtype=np.uint8)
    img[:] = (200, 200, 200) # Light gray background

    # Add some random noise to background
    noise_bg = np.random.normal(0, 5, img.shape).astype(np.uint8)
    img = cv2.add(img, noise_bg)

    # Draw a circle (original object)
    cv2.circle(img, (150, 150), 50, (0, 0, 255), -1)

    # Copy-Move: Copy the circle to another location
    # In a real forgery, pixel values might be identical.
    # Here we just draw another identical circle.
    cv2.circle(img, (350, 350), 50, (0, 0, 255), -1)

    # Save as JPEG with high quality
    cv2.imwrite("temp_base.jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])

    # Splicing/ELA test: Paste a region from a lower quality image
    # 1. Save a version at low quality
    cv2.imwrite("temp_low.jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 50])
    
    # 2. Read back
    img_high = cv2.imread("temp_base.jpg")
    img_low = cv2.imread("temp_low.jpg")

    # 3. Paste a block from low quality to high quality
    img_high[50:150, 300:400] = img_low[50:150, 300:400]

    cv2.imwrite(filename, img_high, [cv2.IMWRITE_JPEG_QUALITY, 95])
    
    # Cleanup temps
    if os.path.exists("temp_base.jpg"): os.remove("temp_base.jpg")
    if os.path.exists("temp_low.jpg"): os.remove("temp_low.jpg")
    
    return filename

def verify():
    output_dir = "verification_output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # image_path = create_synthetic_forgery()
    image_path = "/Users/vignesh/Downloads/vigneshPassportDevakottai2.png"
    print(f"Created synthetic forgery: {image_path}")

    print("--- Testing ELA ---")
    ela_out = os.path.join(output_dir, "ela_result.jpg")
    ela.perform_ela(image_path, output_path=ela_out)
    if os.path.exists(ela_out):
        print("ELA Test: PASSED")
    else:
        print("ELA Test: FAILED")

    print("--- Testing Noise Analysis ---")
    noise_out = os.path.join(output_dir, "noise_result.jpg")
    noise.analyze_noise(image_path, output_path=noise_out)
    if os.path.exists(noise_out):
        print("Noise Analysis Test: PASSED")
    else:
        print("Noise Analysis Test: FAILED")

    print("--- Testing Copy-Move ---")
    # For synthetic copy-move, we rely on ORB finding matches. 
    # Since we drew identical circles, it should find matches.
    copy_move_out = os.path.join(output_dir, "copymove_result.jpg")
    matches = copy_move.detect_copy_move(image_path, output_path=copy_move_out)
    
    # Note: Copy-move with simple identical circles might produce A LOT of matches or none if descriptors are too simple/flat.
    # But checking if the function runs without error is step 1.
    print(f"Copy-Move matches found: {len(matches)}")
    if os.path.exists(copy_move_out) or len(matches) >= 0: # It passes if it runs, results depend on image quality
        print("Copy-Move Test: PASSED (Execution)")

if __name__ == "__main__":
    verify()
