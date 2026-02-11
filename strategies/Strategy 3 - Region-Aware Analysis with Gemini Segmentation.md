# Strategy 3: Region-Aware Analysis with Gemini Segmentation

## Approach

This strategy moves beyond whole-document detection by first breaking the document down into its semantic building blocks. We use the Gemini Vision Model to intelligently segment the document into distinct regions, classifying each as "Photo", "Text", "Seal", "MRZ", or other relevant types.

Once the document is mapped, we apply targeted computer vision analysis appropriate for each specific region type:

1.  **Photo Regions**: Analyzed for lighting consistency, face manipulation artifacts, and background irregularities that might suggest a photo swap.

2.  **Text Fields**: Examined for font consistency, character alignment, and rendering quality to detect if names or dates have been altered.

3.  **MRZ (Machine Readable Zone)**: Validated for checksum correctness and proper formatting.

4.  **Seals and Watermarks**: Checked for edge sharpness, color consistency, and expected placement.

By treating a passport not as a single image but as a collection of distinct components, this method allows us to use the right tool for the right job, reducing false positives caused by applying photo-analysis techniques to text regions or vice versa.
