import cv2
import numpy as np
import pywt
from skimage.metrics import structural_similarity as ssim

import cv2
import numpy as np
import pywt

def embed_watermark(main_image_path, watermark_image_path, output_path, alpha=1):
    # Read the images
    main_image = cv2.imread(main_image_path)
    watermark = cv2.imread(watermark_image_path, cv2.IMREAD_GRAYSCALE)

    # Resize the watermark to match the size of the LL coefficients (1/4 of the main image size)
    watermark = cv2.resize(watermark, (main_image.shape[1] // 4, main_image.shape[0] // 4))

    # Split the main image into its color channels
    b, g, r = cv2.split(main_image)

    # Function to embed watermark in a single channel
    def embed_in_channel(channel):
        coeffs2 = pywt.dwt2(channel, 'haar')
        LL, (LH, HL, HH) = coeffs2

        LL_dct = cv2.dct(np.float32(LL))

        # Embed the watermark in the DCT coefficients
        for i in range(watermark.shape[0]):
            for j in range(watermark.shape[1]):
                if i < LL_dct.shape[0] and j < LL_dct.shape[1]:
                    LL_dct[i, j] *= (1 + alpha * watermark[i, j] / 255)

        # Perform inverse DCT
        LL_watermarked = cv2.idct(LL_dct)

        # Perform inverse DWT
        coeffs2_watermarked = LL_watermarked, (LH, HL, HH)
        return pywt.idwt2(coeffs2_watermarked, 'haar')

    # Embed watermark in each channel
    b_watermarked = embed_in_channel(b)
    g_watermarked = embed_in_channel(g)
    r_watermarked = embed_in_channel(r)

    # Merge the channels back into a color image
    watermarked_image = cv2.merge((b_watermarked, g_watermarked, r_watermarked))

    # Clip and convert back to uint8
    watermarked_image = np.clip(watermarked_image, 0, 255).astype(np.uint8)

    # Save the watermarked image
    cv2.imwrite(output_path, watermarked_image)

# The rest of your code (extract_watermark and compare_images functions) remains the same

def extract_watermark(watermarked_image_path, original_image_path, output_path, alpha=1):
    # Read the images
    watermarked_image = cv2.imread(watermarked_image_path, cv2.IMREAD_GRAYSCALE)
    original_image = cv2.imread(original_image_path, cv2.IMREAD_GRAYSCALE)

    # Ensure both images have the same size
    if watermarked_image.shape != original_image.shape:
        original_image = cv2.resize(original_image, (watermarked_image.shape[1], watermarked_image.shape[0]))

    # Perform DWT on both images
    coeffs2_watermarked = pywt.dwt2(watermarked_image, 'haar')
    LL_watermarked, _ = coeffs2_watermarked

    coeffs2_original = pywt.dwt2(original_image, 'haar')
    LL_original, _ = coeffs2_original

    # Perform DCT on the LL subbands
    LL_watermarked_dct = cv2.dct(np.float32(LL_watermarked))
    LL_original_dct = cv2.dct(np.float32(LL_original))

    # Extract the watermark
    extracted_watermark = np.zeros_like(LL_watermarked_dct)
    for i in range(extracted_watermark.shape[0]):
        for j in range(extracted_watermark.shape[1]):
            if LL_original_dct[i, j] != 0:  # Avoid division by zero
                extracted_watermark[i, j] = (LL_watermarked_dct[i, j] - LL_original_dct[i, j]) / (alpha * LL_original_dct[i, j])

    # Normalize and scale the extracted watermark
    extracted_watermark = (extracted_watermark - np.min(extracted_watermark)) / (np.max(extracted_watermark) - np.min(extracted_watermark))
    extracted_watermark = (extracted_watermark * 255).astype(np.uint8)

    # Resize the extracted watermark to match the original watermark size
    extracted_watermark = cv2.resize(extracted_watermark, (original_image.shape[1] // 4, original_image.shape[0] // 4))

    # Save the extracted watermark
    cv2.imwrite(output_path, extracted_watermark)

def compare_images(original_image_path, watermarked_image_path, threshold=0.95):
    # Load the images in grayscale
    original_image = cv2.imread(original_image_path, cv2.IMREAD_GRAYSCALE)
    watermarked_image = cv2.imread(watermarked_image_path, cv2.IMREAD_GRAYSCALE)

    # Check if the images have the same size
    if original_image.shape != watermarked_image.shape:
        print("Images do not have the same dimensions.")
        return False

    # Calculate SSIM between the two images
    similarity_index, _ = ssim(original_image, watermarked_image, full=True)

    # Print the similarity index
    print(f"Similarity Index: {similarity_index}")

    # Compare with the threshold
    if similarity_index < threshold:
        print("The watermarked image has been altered.")
        return True  # The image is likely watermarked
    else:
        print("The images are similar, no watermark detected.")
        return False  # The images are likely the same

# Example usage
embed_watermark('watermark.jpg', 'Ankur.png', 'new.jpg')
extract_watermark('new.jpg', 'watermark.jpg', 'neee.jpg')

# Check whether watermark has been applied
if compare_images('watermark.jpg', 'neee.jpg'):
    print("The image has been watermarked.")
else:
    print("The image is the same as the original.")
