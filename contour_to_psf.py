import cv2
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev
from skimage.morphology import skeletonize
from skimage.util import invert

PSF_resolution = 800

def find_skeleton(filepath):
    # load
    image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    _, binary_img = cv2.threshold(image, 90, 255, cv2.THRESH_BINARY)
    # binary_img = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
    #          cv2.THRESH_BINARY,11,-2) # This could help with selecting the curve but we'll see
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # largest_contour = max(contours, key=cv2.contourArea)
    largest_contour = contours[0]
    contour_points = largest_contour.squeeze()

    x = contour_points[:, 0]
    y = contour_points[:, 1]

    # fit b-spline to contour and smooth
    tck, u = splprep([x, y], s=30)  # work on making s dependent on x y values for adaptability
    x_smooth, y_smooth = splev(np.linspace(0, 1, PSF_resolution), tck)

    # close and convert spline back to contour
    x_smooth = np.append(x_smooth, x_smooth[0])
    y_smooth = np.append(y_smooth, y_smooth[0])
    smoothed_contour = np.array([np.array([int(x), int(y)]) for x, y in zip(x_smooth, y_smooth)], dtype=np.int32)

    # fill contour & skeletonize in binary
    filled_img = np.zeros_like(binary_img)
    cv2.fillPoly(filled_img, [smoothed_contour], color=255)
    skeleton = skeletonize(filled_img // 255)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.title("Original Image")

    plt.subplot(1, 3, 2)
    plt.imshow(binary_img, cmap='gray')
    plt.plot(x, y, 'r--', label='Original Contour')
    plt.plot(x_smooth, y_smooth, 'b-', label='Spline Curve')
    plt.legend()
    plt.axis('off')
    plt.title("Original and Smoothed Contour")

    plt.subplot(1, 3, 3)
    plt.imshow(skeleton, cmap='gray')
    plt.axis('off')
    plt.title("Skeletonized Smoothed Contour")

    plt.suptitle("Selected Contour to Skeletonized Spline")
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # "artificialcontour.png", "cropped_sample.png", 
    for test_images in ["artificialcontour.png", "cropped_sample.png", "cropped_sample_2.jpg"]:
        find_skeleton(f"assets/{test_images}")