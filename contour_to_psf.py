import cv2
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev
from skimage.morphology import skeletonize
import csv
import os

class Skeletonizer:
    def __init__(self, PSF_resolution=800):
        """
        Initializes the Skeletonizer object with a specified PSF resolution.

        Parameters:
        - PSF_resolution (int): The resolution for the spline interpolation.
        """
        self.PSF_resolution = PSF_resolution
        self.image = None
        self.binary_img = None
        self.contour_points = None
        self.x_smooth = None
        self.y_smooth = None
        self.smoothed_contour = None
        self.skeleton = None
        self.filepath = None

    def load_image(self, filepath):
        """
        Loads an image in grayscale from the given file path.

        Parameters:
        - filepath (str): The path to the image file.

        Returns:
        - success (bool): True if the image was loaded successfully, False otherwise.
        """
        self.filepath = filepath
        self.image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        if self.image is None:
            print(f"Failed to load image: {filepath}")
            return False
        return True

    def threshold_image(self, threshold_value=90):
        """
        Applies binary thresholding to the loaded image.

        Parameters:
        - threshold_value (int): The threshold value for binarization.
        """
        _, self.binary_img = cv2.threshold(self.image, threshold_value, 255, cv2.THRESH_BINARY)

    def find_contours(self):
        """
        Finds contours in the binary image.

        Returns:
        - success (bool): True if contours were found, False otherwise.
        """
        contours, _ = cv2.findContours(self.binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if len(contours) == 0:
            print("No contours found in the image.")
            return False
        # Select the largest contour based on area
        self.largest_contour = max(contours, key=cv2.contourArea)
        self.contour_points = self.largest_contour.squeeze()
        return True

    def fit_spline(self, smoothing_factor=30):
        """
        Fits a B-spline to the contour points.

        Parameters:
        - smoothing_factor (float): The smoothing factor for the spline.

        Returns:
        - success (bool): True if spline fitting was successful, False otherwise.
        """
        x = self.contour_points[:, 0]
        y = self.contour_points[:, 1]
        try:
            tck, u = splprep([x, y], s=smoothing_factor)
        except Exception as e:
            print(f"Error fitting B-spline: {e}")
            return False
        self.x_smooth, self.y_smooth = splev(np.linspace(0, 1, self.PSF_resolution), tck)
        return True

    def create_smoothed_contour(self):
        """
        Closes the spline and converts it back to a contour.
        """
        # Close and convert spline back to contour
        self.x_smooth = np.append(self.x_smooth, self.x_smooth[0])
        self.y_smooth = np.append(self.y_smooth, self.y_smooth[0])
        self.smoothed_contour = np.array([np.array([int(x), int(y)]) for x, y in zip(self.x_smooth, self.y_smooth)], dtype=np.int32)

    def skeletonize_image(self):
        """
        Fills the smoothed contour and applies skeletonization.
        """
        # Fill contour & skeletonize in binary
        filled_img = np.zeros_like(self.binary_img)
        cv2.fillPoly(filled_img, [self.smoothed_contour], color=255)
        self.skeleton = skeletonize(filled_img // 255)

    def save_coords(self, output_dir=None):
        """
        Saves the skeleton coordinates to a CSV file.

        Parameters:
        - output_dir (str): The directory where the CSV file will be saved.
        """
        # Get skeleton coordinates
        skeleton_coords = np.column_stack(np.where(self.skeleton))
        if skeleton_coords.size == 0:
            print("No skeleton found after skeletonization.")
            return
        # Swap columns to have (x, y) format
        skeleton_coords_xy = skeleton_coords[:, [1, 0]]

        # Prepare output directory
        if output_dir is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            image_name = os.path.splitext(os.path.basename(self.filepath))[0]
            output_dir = f"data/output_{image_name}"
            os.makedirs(output_dir, exist_ok=True)

        # Save skeleton coordinates to a CSV file
        skeleton_filename = os.path.join(output_dir, f"skeleton_coords_{image_name}_{timestamp}.csv")
        with open(skeleton_filename, mode='w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['x', 'y'])
            writer.writerows(skeleton_coords_xy)
        print(f"Skeleton coordinates saved to {skeleton_filename}")

    def plot_results(self):
        """
        Plots the original image, contour, spline curve, and skeleton points.
        """
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 3, 1)
        plt.imshow(self.image, cmap='gray')
        plt.axis('off')
        plt.title("Original Image")

        plt.subplot(1, 3, 2)
        plt.imshow(self.binary_img, cmap='gray')
        x = self.contour_points[:, 0]
        y = self.contour_points[:, 1]
        plt.plot(x, y, 'r--', label='Original Contour')
        plt.plot(self.x_smooth, self.y_smooth, 'b-', label='Spline Curve')
        plt.legend()
        plt.axis('off')
        plt.title("Original and Smoothed Contour")

        plt.subplot(1, 3, 3)
        plt.imshow(self.image, cmap='gray')
        skeleton_coords = np.column_stack(np.where(self.skeleton))
        plt.scatter(skeleton_coords[:, 1], skeleton_coords[:, 0], s=1, c='red')
        plt.axis('off')
        plt.title("Skeleton Points on Original Image")

        plt.suptitle("Selected Contour to Skeletonized Spline")
        plt.tight_layout()
        plt.show()

    def process_image(self, filepath, threshold_value=90, smoothing_factor=30):
        """
        Processes the image by performing all the steps: loading, thresholding,
        finding contours, fitting spline, and skeletonizing.

        Parameters:
        - filepath (str): The path to the image file.
        - threshold_value (int): The threshold value for binarization.
        - smoothing_factor (float): The smoothing factor for the spline.
        """
        if not self.load_image(filepath):
            return False
        self.threshold_image(threshold_value)
        if not self.find_contours():
            return False
        if not self.fit_spline(smoothing_factor):
            return False
        self.create_smoothed_contour()
        self.skeletonize_image()
        return True

if __name__ == '__main__':
    test_images = ["artificialcontour.png", "cropped_sample.png", "cropped_sample_2.jpg"]
    for test_image in test_images:
        skeletonizer = Skeletonizer(PSF_resolution=800)
        success = skeletonizer.process_image(f"assets/{test_image}")
        if success:
            skeletonizer.save_coords()
            skeletonizer.plot_results()
