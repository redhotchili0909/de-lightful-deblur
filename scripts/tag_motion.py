import os
import numpy as np
import cv2 as cv
from psf_utils import calculate_psf, visualize_psf, plot_motion_path, smooth_motion_vectors
from skimage.restoration import richardson_lucy, unsupervised_wiener
from pprint import pprint

# Paths
filename = 'op_flow_test5'
datafile_path = f"assets/tag_data/{filename}.npy"
image_path = f"assets/imgs/{filename}.jpg"

# Step 0: Load Motion Vectors
print("Loading motion vectors from data file...")
try:
    centers = np.load(datafile_path)
except FileNotFoundError:
    raise ValueError(f"Data file not found at path: {datafile_path}")


motion_vectors = [centers[i + 1] - centers[i] for i in range(len(centers) - 1)]
# print(f"Loaded {len(motion_vectors)} motion vectors.")

pprint([(i,v) for i,v in enumerate(motion_vectors)])

# Step 1: Smooth Motion Vectors
print("Smoothing motion vectors...")
smoothed_motion_vectors = smooth_motion_vectors(motion_vectors)

# Save motion vector path plot
output_path = f"assets/output/{filename}"
os.makedirs(output_path, exist_ok=True)

smooth_motion_path_filename = f"{output_path}/{filename}_smooth_motion.jpg"
plot_motion_path(smoothed_motion_vectors, save_path=smooth_motion_path_filename)
print(f"Motion path plot saved as: {smooth_motion_path_filename}")

# Step 2: Dynamically Determine PSF Size
print("Determining PSF size...")
psf_size = 15  # Set a fixed size or dynamically calculate based on resolution
print(f"PSF size determined: {psf_size}")

# Step 3: Calculate and Save PSF
print("Calculating PSF...")
psf = calculate_psf(smoothed_motion_vectors, psf_size=psf_size, use_interpolation=True, num_samples=100, no_vert=True)
psf_filename = f"{output_path}/{filename}_psf.jpg"
visualize_psf(psf, save_path=psf_filename)
print(f"PSF heatmap saved as: {psf_filename}")

# Step 4: Load Input Image
print("Loading input image...")
image = cv.imread(image_path)
if image is None:
    raise ValueError(f"Could not load the image at path: {image_path}")

# Convert image to grayscale
image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

# Step 5: Deblur Using Richardson-Lucy
print("Deblurring the image using Richardson-Lucy...")
blurred_normalized = image / 255.0
psf /= np.sum(psf)
deblurred_rl = richardson_lucy(blurred_normalized, psf, num_iter=10)
deblurred_rl = np.clip(deblurred_rl * 255.0, 0, 255).astype(np.uint8)

# Save Richardson-Lucy result
deblurred_rl_filename = f"{output_path}/{filename}_deblurred_rl.jpg"
cv.imwrite(deblurred_rl_filename, deblurred_rl)
print(f"Richardson-Lucy deblurred image saved as: {deblurred_rl_filename}")

# Step 5b: Deblur Using Wiener
print("Deblurring the image using Wiener...")
deblurred_wiener, _ = unsupervised_wiener(blurred_normalized, psf)
deblurred_wiener = np.clip(deblurred_wiener * 255.0, 0, 255).astype(np.uint8)
deblurred_wiener_filename = f"{output_path}/{filename}_deblurred_wiener.jpg"
cv.imwrite(deblurred_wiener_filename, deblurred_wiener)
print(f"Wiener deblurred image saved as: {deblurred_wiener_filename}")
