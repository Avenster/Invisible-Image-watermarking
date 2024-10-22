import cv2
import numpy as np
import pywt

def embed_watermark(main_image_path, watermark_image_path, output_path, alpha=0.1):
    # Read the images
    main_image = cv2.imread(main_image_path, cv2.IMREAD_GRAYSCALE)
    watermark = cv2.imread(watermark_image_path, cv2.IMREAD_GRAYSCALE)
    
    watermark = cv2.resize(watermark, (main_image.shape[1]//2, main_image.shape[0]//2))
    
    coeffs2 = pywt.dwt2(main_image, 'haar')
    LL, (LH, HL, HH) = coeffs2
    
    LL_dct = cv2.dct(np.float32(LL))
    
    for i in range(watermark.shape[0]):
        for j in range(watermark.shape[1]):
            LL_dct[i, j] = LL_dct[i, j] * (1 + alpha * watermark[i, j] / 255)
    
    # Perform inverse DCT
    LL_watermarked = cv2.idct(LL_dct)
    
    # Perform inverse DWT
    coeffs2_watermarked = LL_watermarked, (LH, HL, HH)
    watermarked_image = pywt.idwt2(coeffs2_watermarked, 'haar')
    
    # Clip and convert back to uint8
    watermarked_image = np.clip(watermarked_image, 0, 255).astype(np.uint8)
    
    # Save the watermarked image
    cv2.imwrite(output_path, watermarked_image)

def extract_watermark(watermarked_image_path, original_image_path, output_path, alpha=0.1):
    # Read the images
    watermarked_image = cv2.imread(watermarked_image_path, cv2.IMREAD_GRAYSCALE)
    original_image = cv2.imread(original_image_path, cv2.IMREAD_GRAYSCALE)
    
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
    for i in range(LL_watermarked_dct.shape[0]):
        for j in range(LL_watermarked_dct.shape[1]):
            extracted_watermark[i, j] = (LL_watermarked_dct[i, j] - LL_original_dct[i, j]) / (alpha * LL_original_dct[i, j])
    
    # Normalize and scale the extracted watermark
    extracted_watermark = (extracted_watermark - np.min(extracted_watermark)) / (np.max(extracted_watermark) - np.min(extracted_watermark))
    extracted_watermark = (extracted_watermark * 255).astype(np.uint8)
    
    # Save the extracted watermark
    cv2.imwrite(output_path, extracted_watermark)

# Example usage
embed_watermark('watermark.jpg','14377.jpg', 'watermarked_output.jpg')
extract_watermark('watermarked_output.jpg', '14377.jpg', 'extracted_watermark.jpg')