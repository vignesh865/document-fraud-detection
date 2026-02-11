# Strategy 5: Embedding-Based Region Consistency

## Approach

This strategy treats document verification as a puzzle consistency problem, verifying that all parts of the document "belong" together. We operate on the hypothesis that in a genuine document, all regions—regardless of whether they are text, photo, or background—share subtle visual characteristics (like noise profile, compression artifacts, and lighting) because they were created and processed together.

The workflow involves:

1.  **Region Segmentation**: We break the document into semantic components using Gemini Vision.

2.  **Embedding Extraction**: For each region, we compute a "visual embedding"—a dense numerical vector that represents the region's visual essence—using models like DINOv2 or CLIP. These models are self-supervised and excellent at capturing fine-grained texture and style information without needing specific training on fraud data.

3.  **Consistency Analysis**: We compare the visual embedding of each region against its neighbors and the document as a whole. If a specific region (like a text block or photo) has an embedding that is statistically distant from the rest of the document, it suggests that region may have been pasted from a different source or manipulated in a way that altered its fundamental visual properties.

This method allows us to detect "composite" forgeries (swapped photos, pasted text) without identifying the specific manipulation technique, simply by spotting the "odd piece out."
