import cv2
import numpy as np
import pywt
from skimage.metrics import structural_similarity as ssim
from math import log10

def embed_watermark(main_image_path, watermark_image_path, output_path, alpha=0.5):
    main_image = cv2.imread(main_image_path)
    watermark = cv2.imread(watermark_image_path, cv2.IMREAD_GRAYSCALE)
    watermark = cv2.resize(watermark, (32, 32))
    
    ycrcb = cv2.cvtColor(main_image, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    
    coeffs = pywt.dwt2(y, 'haar')
    LL, (LH, HL, HH) = coeffs
    
    LL_dct = cv2.dct(np.float32(LL))
    
    for i in range(32):
        for j in range(32):
            LL_dct[i+16, j+16] += alpha * watermark[i, j]
    
    LL_watermarked = cv2.idct(LL_dct)
    
    coeffs_watermarked = LL_watermarked, (LH, HL, HH)
    y_watermarked = pywt.idwt2(coeffs_watermarked, 'haar')
    
    ycrcb_watermarked = cv2.merge([y_watermarked.astype(np.uint8), cr, cb])
    watermarked_image = cv2.cvtColor(ycrcb_watermarked, cv2.COLOR_YCrCb2BGR)
    
    cv2.imwrite(output_path, watermarked_image)

def extract_watermark(watermarked_image_path, original_image_path, output_path, alpha=0.5):
    watermarked_image = cv2.imread(watermarked_image_path)
    original_image = cv2.imread(original_image_path)
    
    ycrcb_watermarked = cv2.cvtColor(watermarked_image, cv2.COLOR_BGR2YCrCb)
    ycrcb_original = cv2.cvtColor(original_image, cv2.COLOR_BGR2YCrCb)
    
    y_watermarked = ycrcb_watermarked[:,:,0]
    y_original = ycrcb_original[:,:,0]
    
    coeffs_watermarked = pywt.dwt2(y_watermarked, 'haar')
    LL_watermarked, _ = coeffs_watermarked
    
    coeffs_original = pywt.dwt2(y_original, 'haar')
    LL_original, _ = coeffs_original
    
    LL_watermarked_dct = cv2.dct(np.float32(LL_watermarked))
    LL_original_dct = cv2.dct(np.float32(LL_original))
    
    extracted_watermark = np.zeros((32, 32), dtype=np.float32)
    for i in range(32):
        for j in range(32):
            extracted_watermark[i, j] = (LL_watermarked_dct[i+16, j+16] - LL_original_dct[i+16, j+16]) / alpha
    
    extracted_watermark = (extracted_watermark - np.min(extracted_watermark)) / (np.max(extracted_watermark) - np.min(extracted_watermark))
    extracted_watermark = (extracted_watermark * 255).astype(np.uint8)
    
    cv2.imwrite(output_path, extracted_watermark)

def compare_images(original_image_path, watermarked_image_path, threshold=1):
    original_image = cv2.imread(original_image_path)
    watermarked_image = cv2.imread(watermarked_image_path)
    
    original_gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    watermarked_gray = cv2.cvtColor(watermarked_image, cv2.COLOR_BGR2GRAY)
    
    similarity_index, _ = ssim(original_gray, watermarked_gray, full=True)
    
    print(f"Similarity Index: {similarity_index}")
    
    if similarity_index < threshold:
        print("The watermarked image has been altered.")
        return True
    else:
        print("The images are very similar, watermark is not visibly detectable.")
        return False

def calculate_psnr(original_image_path, watermarked_image_path):
    """
    Calculate Peak Signal-to-Noise Ratio between original and watermarked images
    """
    original = cv2.imread(original_image_path)
    watermarked = cv2.imread(watermarked_image_path)
    
    mse = np.mean((original - watermarked) ** 2)
    if mse == 0:
        return float('inf')
    
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / np.sqrt(mse))
    return psnr

def calculate_ncc(original_image_path, watermarked_image_path):
    """
    Calculate Normalized Cross-Correlation between original and watermarked images
    """
    original = cv2.imread(original_image_path, cv2.IMREAD_GRAYSCALE)
    watermarked = cv2.imread(watermarked_image_path, cv2.IMREAD_GRAYSCALE)
    
    # Convert to float for calculations
    original = original.astype(float)
    watermarked = watermarked.astype(float)
    
    # Calculate means
    original_mean = np.mean(original)
    watermarked_mean = np.mean(watermarked)
    
    # Calculate correlation
    numerator = np.sum((original - original_mean) * (watermarked - watermarked_mean))
    denominator = np.sqrt(np.sum((original - original_mean)**2) * np.sum((watermarked - watermarked_mean)**2))
    
    if denominator == 0:
        return 0
    
    ncc = numerator / denominator
    return ncc

def evaluate_watermark_quality(original_image_path, watermarked_image_path):
    """
    Evaluate the quality of the watermarking using multiple metrics
    """
    psnr_value = calculate_psnr(original_image_path, watermarked_image_path)
    
    ncc_value = calculate_ncc(original_image_path, watermarked_image_path)
    
    # Calculate SSIM
    original_image = cv2.imread(original_image_path)
    watermarked_image = cv2.imread(watermarked_image_path)
    original_gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    watermarked_gray = cv2.cvtColor(watermarked_image, cv2.COLOR_BGR2GRAY)
    ssim_value, _ = ssim(original_gray, watermarked_gray, full=True)
    
    print("\nWatermark Quality Metrics:")
    print(f"PSNR: {psnr_value:.2f} dB")
    print(f"NCC: {ncc_value:.4f}")
    print(f"SSIM: {ssim_value:.4f}")
    print("\nQuality Assessment:")
    if psnr_value > 40:
        print("- PSNR indicates excellent image quality")
    elif psnr_value > 30:
        print("- PSNR indicates good image quality")
    else:
        print("- PSNR indicates visible degradation")
        
    if ncc_value > 0.95:
        print("- NCC indicates strong correlation with original image")
    elif ncc_value > 0.85:
        print("- NCC indicates good correlation with original image")
    else:
        print("- NCC indicates significant differences from original image")
    
    return psnr_value, ncc_value, ssim_value

def main():
    # Embed watermark
    embed_watermark('watermark.jpg', 'Ankur.png', 'watermarked_image.jpg', alpha=0.5)
    
    # Extract watermark
    extract_watermark('watermarked_image.jpg', 'watermark.jpg', 'extracted_watermark.png', alpha=1)
    
    # Compare images and evaluate quality
    is_altered = compare_images('watermark.jpg', 'watermarked_image.jpg', threshold=0.99)
    if is_altered:
        print("The image has been invisibly watermarked.")
    else:
        print("No significant changes detected in the image.")
    
    # Evaluate watermark quality
    psnr, ncc, ssim = evaluate_watermark_quality('watermark.jpg', 'watermarked_image.jpg')

if __name__ == "__main__":
    main()