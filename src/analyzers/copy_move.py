
"""
Copy-Move Detection Module
Detects cloned regions within an image using feature matching (SIFT/ORB).
It finds keypoints and descriptors, then matches them to find identical regions.
"""

import cv2
import numpy as np

def detect_copy_move(image_path, output_path=None, min_matches=10):
    """
    Detects copy-move forgery using ORB feature matching.
    
    Args:
        image_path (str): Path to the source image.
        output_path (str, optional): Path to save the visualization of matches.
        min_matches (int): Minimum number of matches to consider as potential forgery.

    Returns:
        list: A list of matched keypoint pairs (points).
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error loading image {image_path}")
        return []

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Initialize ORB detector
    orb = cv2.ORB_create()

    # Find keypoints and descriptors
    keypoints, descriptors = orb.detectAndCompute(gray, None)

    if descriptors is None or len(descriptors) < 2:
        return []

    # Use Brute-Force Matcher with Hamming distance (suitable for binary descriptors like ORB)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False) # CrossCheck=False to allow multiple matches

    # KNN Match
    matches = bf.knnMatch(descriptors, descriptors, k=2) # Find 2 nearest neighbors for each descriptor

    good_matches = []
    # Ratio test as per Lowe's paper
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            # Check if the match is not the point itself (distance > 0)
            # improved check: ensure distance is not 0 
            if m.distance > 0:
                good_matches.append(m)

    # Filter matches based on spatial distance to avoid matching adjacent points
    filtered_matches = []
    min_dist = 50 # Minimum distance in pixels to consider as separate regions
    
    for m in good_matches:
        pt1 = keypoints[m.queryIdx].pt
        pt2 = keypoints[m.trainIdx].pt
        
        dist = np.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)
        
        if dist > min_dist:
             filtered_matches.append(m)

    if output_path:
        # Draw matches
        img_matches = cv2.drawMatches(image, keypoints, image, keypoints, filtered_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv2.imwrite(output_path, img_matches)
        print(f"Copy-Move detection result saved to {output_path}")

    return filtered_matches

