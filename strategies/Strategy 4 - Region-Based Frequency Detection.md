# Strategy 4: Region-Based Frequency Domain Detection

## Approach

This strategy focuses on identifying AI-generated content or deepfakes by analyzing the invisible statistical fingerprints left by generative models. We combine semantic segmentation with specialized frequency-domain detectors to find anomalies that are not visible to the naked eye.

The process follows a three-step workflow:

1.  **Semantic Segmentation**: Using the same method as Strategy 3, we first map the document to identify specific regions like photos, seals, and text blocks.

2.  **Frequency Transformation**: We convert each region from the spatial domain (pixels) to the frequency domain using Fourier transforms. This reveals repeating patterns and spectral artifacts that often characterize generated images.

3.  **Specialized Detection**: We apply two types of detectors to the frequency data:
    *   **GAN Detectors**: These look for "checkerboard" upsampling artifacts and spectral anomalies common in images created by Generative Adversarial Networks (GANs).
    *   **Diffusion Detectors**: These analyze noise residuals and texture patterns to identify signatures specific to diffusion models.

By distinguishing between region types, we can selectively apply these heavy-duty forensic tools only where they matter most—primarily on the photo portrait—avoiding false positives from text or logos where such frequency patterns might occur naturally.
