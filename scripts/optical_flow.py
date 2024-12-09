import os
import cv2 as cv
import numpy as np
from process_video import extract_motion_vectors, determine_kernel_size, visualize_optical_flow
from psf_utils import calculate_psf, visualize_psf, plot_motion_path, smooth_motion_vectors
from skimage.restoration import richardson_lucy

video_path = "assets/vids/op_flow_test1.h264"
image_path = "assets/imgs/op_flow_test1.jpg"


# Step 1: Visualize and save optical flow
center_weight = 0.2
print("Visualizing optical flow...")
optical_flow_image = visualize_optical_flow(video_path, center_weight=center_weight, save_frame=True)

vid_name = os.path.basename(video_path)
vid_base_name, _ = os.path.splitext(vid_name)

output_path = f"assets/output/{vid_base_name}"

os.makedirs(output_path, exist_ok=True)

optical_flow_filename = f"{output_path}/{vid_base_name}_of.jpg"

cv.imwrite(optical_flow_filename, optical_flow_image)
print(f"Optical flow visualization saved as: {optical_flow_filename}")

# Step 2: Extract and smooth motion vectors
print("Extracting motion vectors...")
motion_vectors = extract_motion_vectors(video_path, center_weight=center_weight)
print("Smoothing motion vectors...")
smoothed_motion_vectors = smooth_motion_vectors(motion_vectors)

# Save motion vector path plot
smooth_motion_path_filename = f"{output_path}/{vid_base_name}_smooth_motion.jpg"

plot_motion_path(smoothed_motion_vectors, save_path=smooth_motion_path_filename)
print(f"Motion path plot saved as: {smooth_motion_path_filename}")

# Step 3: Dynamically determine PSF size based on video resolution
print("Determining PSF size...")
psf_size = determine_kernel_size(video_path, scale=0.01, min_size=15, max_size=50)
print(f"PSF size determined: {psf_size}")

# Step 4: Calculate and save the PSF
print("Calculating PSF...")
psf = calculate_psf(smoothed_motion_vectors, psf_size=psf_size, use_interpolation=True, num_samples=100, no_vert=False)
psf_filename = f"{output_path}/{vid_base_name}_psf.jpg"
visualize_psf(psf, save_path=psf_filename)
print(f"PSF heatmap saved as: {psf_filename}")

# Step 5: Load the input image
print("Loading input image...")
image = cv.imread(image_path)
if image is None:
    raise ValueError(f"Could not load the image at path: {image_path}")

# Convert the image to grayscale for processing
image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

# Step 6: Deblur the image using Richardson-Lucy
print("Deblurring the image...")
blurred_normalized = image / 255.0
psf /= np.sum(psf)
deblurred = richardson_lucy(blurred_normalized, psf, num_iter=30)
deblurred = np.clip(deblurred * 255.0, 0, 255).astype(np.uint8)

# Step 7: Save results with dynamic filenames
image_name = os.path.basename(image_path)
image_base_name, _ = os.path.splitext(image_name)
deblurred_filename = f"{output_path}/{image_base_name}_deblurred.jpg"
cv.imwrite(deblurred_filename, deblurred)
print(f"Deblurred image saved as: {deblurred_filename}")
