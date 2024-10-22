import cv2
import numpy as np
import pywt

# Load the images
watermarked_image_path = 'new.jpg'
original_image_path = 'watermark.jpg'
watermarked_image = cv2.imread(watermarked_image_path)
original_image = cv2.imread(original_image_path)

# Define watermark shape based on the original image
watermark_shape = (original_image.shape[0] // 4, original_image.shape[1] // 4)

# Split the images into their color channels
watermarked_blue, watermarked_green, watermarked_red = cv2.split(watermarked_image)
original_blue, original_green, original_red = cv2.split(original_image)

# Function to extract watermark from a single channel
def extract_from_channel(watermarked_channel, original_channel, wm_height, wm_width, alpha=0.05):
    # Apply DWT to both watermarked and original channel
    coeffs_watermarked = pywt.dwt2(watermarked_channel, 'haar')
    LL_w, (LH_w, HL_w, HH_w) = coeffs_watermarked

    coeffs_original = pywt.dwt2(original_channel, 'haar')
    LL_o, (LH_o, HL_o, HH_o) = coeffs_original

    # Extract the watermark by subtracting the HH components
    extracted_watermark = (HH_w[:wm_height, :wm_width] - HH_o[:wm_height, :wm_width]) / alpha

    # Convert extracted watermark back to original scale
    extracted_watermark = np.clip(extracted_watermark * 255, 0, 255).astype(np.uint8)
    return extracted_watermark

# Extract watermark from each color channel
extracted_blue_wm = extract_from_channel(watermarked_blue, original_blue, watermark_shape[0], watermark_shape[1])
extracted_green_wm = extract_from_channel(watermarked_green, original_green, watermark_shape[0], watermark_shape[1])
extracted_red_wm = extract_from_channel(watermarked_red, original_red, watermark_shape[0], watermark_shape[1])

# Average the extracted watermarks from the color channels
extracted_watermark = (extracted_blue_wm + extracted_green_wm + extracted_red_wm) // 3

# Save the extracted watermark for comparison
output_extracted_watermark_path = 'extracted_watermark_adjusted2.png'
cv2.imwrite(output_extracted_watermark_path, extracted_watermark)

output_extracted_watermark_path
