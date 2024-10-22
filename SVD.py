import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

def embed_watermark(main_image_path, watermark_image_path, output_path, alpha=1):
    # Read images
    main_image = cv2.imread(main_image_path)
    if main_image is None:
        raise ValueError(f"Could not read main image: {main_image_path}")
        
    watermark = cv2.imread(watermark_image_path, cv2.IMREAD_GRAYSCALE)
    if watermark is None:
        raise ValueError(f"Could not read watermark image: {watermark_image_path}")
    
    # Get dimensions of main image
    height, width = main_image.shape[:2]
    
    # Calculate number of complete blocks
    blocks_height = height // 32
    blocks_width = width // 32
    
    # Resize watermark to match the number of blocks
    watermark = cv2.resize(watermark, (blocks_width, blocks_height))
    
    # Convert to YCrCb
    ycrcb = cv2.cvtColor(main_image, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    
    # Process only complete blocks
    block_size = 32
    for i in range(blocks_height):
        for j in range(blocks_width):
            # Get current block coordinates
            y_start = i * block_size
            y_end = (i + 1) * block_size
            x_start = j * block_size
            x_end = (j + 1) * block_size
            
            block = y[y_start:y_end, x_start:x_end]
            
            # Apply SVD to each block
            U, S, Vt = np.linalg.svd(block, full_matrices=True)
            
            # Modify singular values with watermark
            S_modified = S.copy()
            S_modified[0] += alpha * watermark[i, j]
            
            # Reconstruct block
            block_watermarked = np.dot(U, np.dot(np.diag(S_modified), Vt))
            y[y_start:y_end, x_start:x_end] = block_watermarked
    
    # Ensure pixel values are within valid range
    y = np.clip(y, 0, 255)
    
    # Merge channels and convert back to BGR
    ycrcb_watermarked = cv2.merge([y.astype(np.uint8), cr, cb])
    watermarked_image = cv2.cvtColor(ycrcb_watermarked, cv2.COLOR_YCrCb2BGR)
    
    cv2.imwrite(output_path, watermarked_image)
    return blocks_height, blocks_width

def extract_watermark(watermarked_image_path, original_image_path, output_path, alpha=1):
    # Read images
    watermarked_image = cv2.imread(watermarked_image_path)
    if watermarked_image is None:
        raise ValueError(f"Could not read watermarked image: {watermarked_image_path}")
        
    original_image = cv2.imread(original_image_path)
    if original_image is None:
        raise ValueError(f"Could not read original image: {original_image_path}")
    
    # Convert to YCrCb
    ycrcb_watermarked = cv2.cvtColor(watermarked_image, cv2.COLOR_BGR2YCrCb)
    ycrcb_original = cv2.cvtColor(original_image, cv2.COLOR_BGR2YCrCb)
    
    y_watermarked = ycrcb_watermarked[:,:,0]
    y_original = ycrcb_original[:,:,0]
    
    # Calculate dimensions
    height, width = y_watermarked.shape
    blocks_height = height // 32
    blocks_width = width // 32
    block_size = 32
    
    # Initialize extracted watermark
    extracted_watermark = np.zeros((blocks_height, blocks_width), dtype=np.float32)
    
    # Extract watermark from each block
    for i in range(blocks_height):
        for j in range(blocks_width):
            y_start = i * block_size
            y_end = (i + 1) * block_size
            x_start = j * block_size
            x_end = (j + 1) * block_size
            
            block_watermarked = y_watermarked[y_start:y_end, x_start:x_end]
            block_original = y_original[y_start:y_end, x_start:x_end]
            
            # Apply SVD to both blocks
            _, S_watermarked, _ = np.linalg.svd(block_watermarked, full_matrices=True)
            _, S_original, _ = np.linalg.svd(block_original, full_matrices=True)
            
            # Extract watermark from singular values
            extracted_watermark[i, j] = (S_watermarked[0] - S_original[0]) / alpha
    
    # Normalize the extracted watermark
    extracted_watermark = extracted_watermark - np.min(extracted_watermark)
    extracted_watermark = (extracted_watermark / np.max(extracted_watermark) * 255)
    
    # Post-processing to improve visibility
    extracted_watermark = extracted_watermark.astype(np.uint8)
    extracted_watermark = cv2.GaussianBlur(extracted_watermark, (3,3), 0)
    extracted_watermark = cv2.equalizeHist(extracted_watermark)
    
    cv2.imwrite(output_path, extracted_watermark)

def compare_images(original_image_path, watermarked_image_path, threshold=0.98):
    original_image = cv2.imread(original_image_path)
    watermarked_image = cv2.imread(watermarked_image_path)
    
    if original_image is None or watermarked_image is None:
        raise ValueError("Could not read one or both images for comparison")
    
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

def main():
    # Embedding parameters
    alpha = 0.1  # Watermark strength
    
    try:
        # Embed watermark
        print("Embedding watermark...")
        blocks_height, blocks_width = embed_watermark('watermark.jpg', 'Vector.png', 'watermarked_image.jpg', alpha=alpha)
        print(f"Watermark embedded. Image divided into {blocks_height}x{blocks_width} blocks")
        
        # Extract watermark
        print("\nExtracting watermark...")
        extract_watermark('watermarked_image.jpg', 'watermark.jpg', 'extracted_watermark.png', alpha=alpha)
        print("Watermark extracted")
        
        # Compare images
        print("\nComparing images...")
        compare_images('watermark.jpg', 'watermarked_image.jpg', threshold=0.95)
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()