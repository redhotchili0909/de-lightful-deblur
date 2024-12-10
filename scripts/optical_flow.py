import os
import cv2 as cv
import numpy as np
from process_video import *
from psf_utils import *
from skimage.restoration import richardson_lucy, unsupervised_wiener

filename = 'op_flow_test1'

video_path = f"assets/vids/{filename}.h264"
image_path = f"assets/imgs/{filename}.jpg"

# Step 0: Manually Select ROI
print("Manually selecting ROI...")
first_frame = cv.VideoCapture(video_path).read()[1]
if first_frame is None:
    raise ValueError(f"Could not read the first frame from the video at path: {video_path}")

roi = cv.selectROI("Select ROI", first_frame, fromCenter=False, showCrosshair=True)
cv.destroyAllWindows()
if roi == (0, 0, 0, 0):
    raise ValueError("No ROI was selected.")

roi = [int(roi[0]), int(roi[1]), int(roi[2]), int(roi[3])]
print(f"Selected ROI: {roi}")

# Step 1: Visualize and save optical flow
print("Visualizing optical flow...")
optical_flow_image = visualize_optical_flow(video_path, roi=roi, save_frame=True)

vid_name = os.path.basename(video_path)
vid_base_name, _ = os.path.splitext(vid_name)

output_path = f"assets/output/{vid_base_name}"
os.makedirs(output_path, exist_ok=True)

optical_flow_filename = f"{output_path}/{vid_base_name}_of.jpg"
cv.imwrite(optical_flow_filename, optical_flow_image)
print(f"Optical flow visualization saved as: {optical_flow_filename}")

# Step 2: Extract and smooth motion vectors
print("Extracting motion vectors...")
motion_vectors = extract_motion_vectors(video_path, roi=roi)

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
psf = calculate_psf(smoothed_motion_vectors, psf_size=psf_size, use_interpolation=True, num_samples=100, no_vert=True)
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
print("Deblurring the image using Richardson-Lucy...")
blurred_normalized = image / 255.0
psf /= np.sum(psf)
deblurred_rl = richardson_lucy(blurred_normalized, psf, num_iter=10)
deblurred_rl = np.clip(deblurred_rl * 255.0, 0, 255).astype(np.uint8)

# Step 6b: Deblur the image using Wiener
print("Deblurring the image using Wiener...")
deblurred_wiener, _ = unsupervised_wiener(blurred_normalized, psf)
deblurred_wiener = np.clip(deblurred_wiener * 255.0, 0, 255).astype(np.uint8)

# Step 7: Save results with dynamic filenames
image_name = os.path.basename(image_path)
image_base_name, _ = os.path.splitext(image_name)

# Save Richardson-Lucy result
deblurred_rl_filename = f"{output_path}/{image_base_name}_deblurred_rl.jpg"
cv.imwrite(deblurred_rl_filename, deblurred_rl)
print(f"Richardson-Lucy deblurred image saved as: {deblurred_rl_filename}")

# Save Wiener result
deblurred_wiener_filename = f"{output_path}/{image_base_name}_deblurred_wiener.jpg"
cv.imwrite(deblurred_wiener_filename, deblurred_wiener)
print(f"Wiener deblurred image saved as: {deblurred_wiener_filename}")
