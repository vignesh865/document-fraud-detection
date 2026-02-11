# Strategy 2: Traditional Computer Vision Analysis

## Approach

This strategy applies classical computer vision techniques to the entire document image without relying on AI models or external APIs. We process the image through a suite of independent analyzers, each designed to detect specific mathematical or statistical artifacts that often result from digital manipulation.

The core analyzers include:

1.  **Error Level Analysis (ELA)**: This technique re-saves the image at a known compression level and computes the difference between the original and the re-saved version. It highlights regions that have different compression histories, which is a common indicator of "splicing" (where a portion of one image is pasted into another).

2.  **Noise Variance Analysis**: Digital camera sensors produce a unique, uniform noise pattern across an image. This analyzer examines local noise variance to identify regions where the noise pattern is inconsistent, suggesting that the content may have originated from a different source or camera.

3.  **Font Consistency**: This method extracts text regions and analyzes stroke width, spacing, and aliasing characteristics. It aims to detect characters or words that have been digitally overlaid or altered, as these often fail to match the precise rendering properties of the original document text.

4.  **Resolution Analysis**: This checks for sharp discontinuities in image sharpness or quality. Pasting a high-resolution photo into a lower-resolution document (or vice versa) often leaves tell-tale edge artifacts that this analyzer seeks to identify.

Each component produces an independent technical assessment, and these signals are aggregated to form a conclusion about the document's integrity.
