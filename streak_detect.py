import os
import cv2
import numpy as np
from datetime import datetime

def detect(image, output_subdir, clip_limit=2.0, threshold_value=130, min_contour_area=80, min_brightness=150):
    """
    Detects and numbers bright, thick light streaks in an image, saving each streak's close-up image.

    Parameters:
    - image (ndarray or str): Path to the input image or a color image array.
    - output_subdir (str): Directory to save zoomed-in contour images.
    - clip_limit (float): Clip limit for CLAHE contrast enhancement.
    - threshold_value (int): Threshold value for binarization.
    - min_contour_area (int): Minimum area for detected contours to be considered streaks.
    - min_brightness (int): Minimum average brightness inside a contour to prioritize whiter contours.

    Returns:
    - output_image (ndarray): Color image with detected streaks outlined and numbered.
    - contours (list): List of contours representing detected streaks.
    """

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Load the image in color and grayscale for processing
    if isinstance(image, str):
        color_image = cv2.imread(image, cv2.IMREAD_COLOR)
        grayscale_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    else:
        if len(image.shape) == 2:
            # Grayscale image
            grayscale_image = image
            color_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif len(image.shape) == 3 and image.shape[2] == 3:
            # Color image
            color_image = image
            grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            raise ValueError("Invalid image format.")
    
    if grayscale_image is None:
        raise ValueError("Image not found or invalid image path.")
    
    # Enhance contrast using CLAHE
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    enhanced_image = clahe.apply(grayscale_image)
    
    # Threshold the image to isolate bright regions
    _, binary_image = cv2.threshold(enhanced_image, threshold_value, 255, cv2.THRESH_BINARY)
    
    # Morphological filtering to clean up noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    cleaned_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(cleaned_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Use the color image for output
    output_image = color_image.copy()
    os.makedirs(output_subdir, exist_ok=True)
    
    contour_count = 1
    for contour in contours:
        if cv2.contourArea(contour) > min_contour_area:
            # Calculate mean brightness within the contour
            mask = np.zeros(grayscale_image.shape, dtype=np.uint8)
            cv2.drawContours(mask, [contour], -1, 255, -1)
            mean_brightness = cv2.mean(grayscale_image, mask=mask)[0]
            
            if mean_brightness >= min_brightness:
                # Draw the contour outline on the output image
                cv2.drawContours(output_image, [contour], -1, (0, 0, 255), 2)
                
                # Number each contour with offset
                x, y, w, h = cv2.boundingRect(contour)
                text_x = x
                text_y = y - 10
                if text_y < 0:
                    text_y = y + h + 20
                cv2.putText(output_image, str(contour_count), (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.8, (0, 0, 255), 2)
        
                # Save zoomed-in image of the contour
                x, y, w, h = cv2.boundingRect(contour)
                margin = 100
                x1, y1, x2, y2 = max(0, x - margin), max(0, y - margin), min(grayscale_image.shape[1], x + w + margin), min(grayscale_image.shape[0], y + h + margin)
                zoomed_contour = output_image[y1:y2, x1:x2]
                zoomed_filename = os.path.join(output_subdir, f"streak_{contour_count}_{timestamp}.png")
                cv2.imwrite(zoomed_filename, zoomed_contour)
                
                contour_count += 1
    
    # Save the annotated full image in color
    full_image_filename = os.path.join(output_subdir, f"processed_full_image_{timestamp}.png")
    cv2.imwrite(full_image_filename, output_image)

    return output_image, contours


def process_all_images(input_folder="assets", output_folder="auto_select", clip_limit=2.0, threshold_value=130, min_contour_area=80, min_brightness=150):
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Iterate through each image in the input folder
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(input_folder, filename)
            image_output_subdir = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_output")
            
            # Process image and save results
            output_image, contours = detect(image_path, image_output_subdir, clip_limit=clip_limit, 
                                            threshold_value=threshold_value, min_contour_area=min_contour_area, 
                                            min_brightness=min_brightness)
            print(f"Processed and saved images in: {image_output_subdir}")

if __name__ == '__main__':
    import time

    start = time.time()
    process_all_images()
    end = time.time()

    print('Processing took %.2f seconds' % (end - start))
