import cv2
import numpy as np
import pywt
from skimage.metrics import structural_similarity as ssim

def embed_watermark(main_image_path, watermark_image_path, output_path, alpha=0.1):
    # Read the images
    main_image = cv2.imread(main_image_path)
    watermark = cv2.imread(watermark_image_path, cv2.IMREAD_GRAYSCALE)

    # Resize watermark to a fixed size
    watermark = cv2.resize(watermark, (32, 32))

    # Convert main image to YCrCb color space
    ycrcb = cv2.cvtColor(main_image, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)

    # Apply DWT to Y channel
    coeffs = pywt.dwt2(y, 'haar')
    LL, (LH, HL, HH) = coeffs

    # Apply DCT to LL subband
    LL_dct = cv2.dct(np.float32(LL))

    # Embed watermark
    for i in range(32):
        for j in range(32):
            LL_dct[i, j] += alpha * watermark[i, j]

    # Apply inverse DCT
    LL_watermarked = cv2.idct(LL_dct)

    # Apply inverse DWT
    coeffs_watermarked = LL_watermarked, (LH, HL, HH)
    y_watermarked = pywt.idwt2(coeffs_watermarked, 'haar')

    # Merge channels and convert back to BGR
    ycrcb_watermarked = cv2.merge([y_watermarked.astype(np.uint8), cr, cb])
    watermarked_image = cv2.cvtColor(ycrcb_watermarked, cv2.COLOR_YCrCb2BGR)

    # Save the watermarked image
    cv2.imwrite(output_path, watermarked_image)

def extract_watermark(watermarked_image_path, original_image_path, output_path, alpha=0.1):
    # Read the images
    watermarked_image = cv2.imread(watermarked_image_path)
    original_image = cv2.imread(original_image_path)

    # Convert images to YCrCb color space
    ycrcb_watermarked = cv2.cvtColor(watermarked_image, cv2.COLOR_BGR2YCrCb)
    ycrcb_original = cv2.cvtColor(original_image, cv2.COLOR_BGR2YCrCb)

    # Split channels
    y_watermarked = ycrcb_watermarked[:,:,0]
    y_original = ycrcb_original[:,:,0]

    # Apply DWT to both Y channels
    coeffs_watermarked = pywt.dwt2(y_watermarked, 'haar')
    LL_watermarked, _ = coeffs_watermarked

    coeffs_original = pywt.dwt2(y_original, 'haar')
    LL_original, _ = coeffs_original

    # Apply DCT to both LL subbands
    LL_watermarked_dct = cv2.dct(np.float32(LL_watermarked))
    LL_original_dct = cv2.dct(np.float32(LL_original))

    # Extract watermark
    extracted_watermark = np.zeros((32, 32), dtype=np.float32)
    for i in range(32):
        for j in range(32):
            extracted_watermark[i, j] = (LL_watermarked_dct[i, j] - LL_original_dct[i, j]) / alpha

    # Normalize and scale the extracted watermark
    extracted_watermark = (extracted_watermark - np.min(extracted_watermark)) / (np.max(extracted_watermark) - np.min(extracted_watermark))
    extracted_watermark = (extracted_watermark * 255).astype(np.uint8)

    # Save the extracted watermark
    cv2.imwrite(output_path, extracted_watermark)

def compare_images(original_image_path, watermarked_image_path, threshold=0.98):
    # Load the images
    original_image = cv2.imread(original_image_path)
    watermarked_image = cv2.imread(watermarked_image_path)

    # Convert images to grayscale
    original_gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    watermarked_gray = cv2.cvtColor(watermarked_image, cv2.COLOR_BGR2GRAY)

    # Calculate SSIM between the two images
    similarity_index, _ = ssim(original_gray, watermarked_gray, full=True)

    # Print the similarity index
    print(f"Similarity Index: {similarity_index}")

    # Compare with the threshold
    if similarity_index < threshold:
        print("The watermarked image has been altered.")
        return True  # The image is likely watermarked
    else:
        print("The images are very similar, watermark is not visibly detectable.")
        return False  # The images are visually the same

# Example usage
embed_watermark('watermark.jpg', 'Ankur.png', 'watermarked_image.jpg', alpha=0.1)
extract_watermark('watermarked_image.jpg', 'watermark.jpg', 'extracted_watermark.png', alpha=0.1)

# Check whether watermark has been applied
if compare_images('watermark.jpg', 'watermarked_image.jpg', threshold=0.99):
    print("The image has been invisibly watermarked.")
else:
    print("No significant changes detected in the image.")