import cv2
import numpy as np
import pywt

def embed_watermark(main_image_path, watermark_image_path, output_image_path):
    # Load main image in color (3 channels) and watermark in grayscale
    main_image = cv2.imread(main_image_path)  # Load the main image in color
    watermark = cv2.imread(watermark_image_path, cv2.IMREAD_GRAYSCALE)

    wm_height, wm_width = main_image.shape[0] // 4, main_image.shape[1] // 4
    watermark_resized = cv2.resize(watermark, (wm_width, wm_height))

    # Split the main image into its color channels (B, G, R)
    blue_channel, green_channel, red_channel = cv2.split(main_image)

    # Function to embed watermark into a single channel
    def embed_in_channel(channel, watermark_resized, alpha=1):
        # Apply DWT on the channel
        coeffs = pywt.dwt2(channel, 'haar')
        LL, (LH, HL, HH) = coeffs

        # Embed watermark into the HH component
        wm_resized_float = watermark_resized.astype(np.float32) / 255
        HH[:wm_height, :wm_width] += wm_resized_float * alpha  # Adjust alpha for invisibility

        # Perform inverse DWT
        coeffs_embedded = (LL, (LH, HL, HH))
        channel_embedded = pywt.idwt2(coeffs_embedded, 'haar')

        return np.clip(channel_embedded, 0, 255).astype(np.uint8)

    # Embed watermark into each color channel
    blue_channel_embedded = embed_in_channel(blue_channel, watermark_resized)
    green_channel_embedded = embed_in_channel(green_channel, watermark_resized)
    red_channel_embedded = embed_in_channel(red_channel, watermark_resized)

    # Merge the channels back into a color image
    watermarked_image = cv2.merge((blue_channel_embedded, green_channel_embedded, red_channel_embedded))

    # Save the watermarked image
    cv2.imwrite(output_image_path, watermarked_image)

    print(f"Watermark embedded and saved to {output_image_path}")

# Usage example
main_image_path = 'watermark.jpg'  # Path to the main image
watermark_image_path = 'Vector.png'  # Path to the watermark image
output_image_path = 'wh.jpg'  # Path where the watermarked image will be saved

embed_watermark(main_image_path, watermark_image_path, output_image_path)
