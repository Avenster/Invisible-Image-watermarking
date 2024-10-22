import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

def embed_watermark(main_image_path, watermark_image_path, output_path, alpha=1):
    main_image = cv2.imread(main_image_path)
    watermark = cv2.imread(watermark_image_path, cv2.IMREAD_GRAYSCALE)
    watermark = cv2.resize(watermark, (32, 32))

    ycrcb = cv2.cvtColor(main_image, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)

    U, S, Vt = np.linalg.svd(y, full_matrices=False)

    for i in range(32):
        for j in range(32):
            S[i] += alpha * watermark[i, j]

    y_watermarked = np.dot(U, np.dot(np.diag(S), Vt))

    ycrcb_watermarked = cv2.merge([y_watermarked.astype(np.uint8), cr, cb])
    watermarked_image = cv2.cvtColor(ycrcb_watermarked, cv2.COLOR_YCrCb2BGR)

    cv2.imwrite(output_path, watermarked_image)

def extract_watermark(watermarked_image_path, original_image_path, output_path, alpha=1):
    watermarked_image = cv2.imread(watermarked_image_path)
    original_image = cv2.imread(original_image_path)

    ycrcb_watermarked = cv2.cvtColor(watermarked_image, cv2.COLOR_BGR2YCrCb)
    ycrcb_original = cv2.cvtColor(original_image, cv2.COLOR_BGR2YCrCb)
    y_watermarked = ycrcb_watermarked[:,:,0]
    y_original = ycrcb_original[:,:,0]

    _, S_watermarked, _ = np.linalg.svd(y_watermarked, full_matrices=False)
    _, S_original, _ = np.linalg.svd(y_original, full_matrices=False)

    extracted_watermark = np.zeros((32, 32), dtype=np.float32)
    for i in range(32):
        for j in range(32):
            extracted_watermark[i, j] = (S_watermarked[i] - S_original[i]) / alpha

    extracted_watermark = (extracted_watermark - np.min(extracted_watermark)) / (np.max(extracted_watermark) - np.min(extracted_watermark))
    extracted_watermark = (extracted_watermark * 255).astype(np.uint8)

    cv2.imwrite(output_path, extracted_watermark)

def compare_images(original_image_path, watermarked_image_path, threshold=0.98):
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

# Example usage
embed_watermark('watermark.jpg', 'Ankur.png', 'watermarked_image.jpg', alpha=0.2)
extract_watermark('watermarked_image.jpg', 'watermark.jpg', 'extracted_watermark.png', alpha=1)

# Check whether watermark has been applied
if compare_images('watermark.jpg', 'watermarked_image.jpg', threshold=1):
    print("The image has been invisibly watermarked.")
else:
    print("No significant changes detected in the image.")