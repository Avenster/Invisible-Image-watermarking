import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

def compare_watermarks(original_watermark_path, extracted_watermark_path):
    # Read the original and extracted watermarks
    original_watermark = cv2.imread(original_watermark_path, cv2.IMREAD_GRAYSCALE)
    extracted_watermark = cv2.imread(extracted_watermark_path, cv2.IMREAD_GRAYSCALE)

    # Ensure both images have the same size
    if original_watermark.shape != extracted_watermark.shape:
        extracted_watermark = cv2.resize(extracted_watermark, (original_watermark.shape[1], original_watermark.shape[0]))

    # Calculate SSIM
    similarity_index, _ = ssim(original_watermark, extracted_watermark, full=True)
    print(f"Similarity Index between original and extracted watermark: {similarity_index}")

# Usage
compare_watermarks('Ankur.png', 'extracted_watermark_adjusted2.png')  # Assuming 'neee.jpg' is the extracted watermark
