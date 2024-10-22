import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

def compare_images(original_image_path, watermarked_image_path, threshold=0.99):
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

# Usage example
original_image_path = 'watermark.jpg'  # Path to the original image
watermarked_image_path = 'new.jpg'  # Path to the watermarked image

if compare_images(original_image_path, watermarked_image_path):
    print("The image has been watermarked.")
else:
    print("The image is the same as the original.")
