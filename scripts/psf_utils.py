import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter1d


def calculate_psf(motion_vectors, psf_size, use_interpolation=False, num_samples=100, no_vert = False):
    """
    Calculate the Point Spread Function (PSF) from motion vectors.

    Parameters:
        motion_vectors (list of tuples): A list of (dx, dy) motion vectors.
        psf_size (int): Size of the PSF kernel (must be odd).
        use_interpolation (bool): Whether to interpolate the motion path.
        num_samples (int): Number of interpolation points (if using interpolation).
        no_vert (bool): Whether to zero out verical motion

    Returns:
        psf (numpy.ndarray): A 2D PSF kernel normalized to sum to 1.
    """
    dx = [vec[0] for vec in motion_vectors]

    if no_vert == True:
        dy = [0] * len(motion_vectors)
    else:
        dy = [vec[1] for vec in motion_vectors]

    x_path = np.cumsum(dx)
    y_path = np.cumsum(dy)

    if use_interpolation:
        t = np.linspace(0, 1, len(x_path))
        t_interp = np.linspace(0, 1, num_samples)
        spline_x = CubicSpline(t, x_path)
        spline_y = CubicSpline(t, y_path)
        x_path = spline_x(t_interp)
        y_path = spline_y(t_interp)

    # Normalize motion path to fit PSF size
    x_path -= np.min(x_path)
    x_path = (x_path / np.max(x_path)) * (psf_size - 1)
    y_path -= np.min(y_path)

    # Avoid division by zero when normalizing y_path
    if np.max(y_path) > 0:
        y_path = (y_path / np.max(y_path)) * (psf_size - 1)

    psf = np.zeros((psf_size, psf_size))
    for x, y in zip(x_path, y_path):
        x_clipped = int(np.clip(x, 0, psf_size - 1))
        y_clipped = int(np.clip(y, 0, psf_size - 1))
        psf[y_clipped, x_clipped] += 1

    if np.sum(psf) == 0:
        raise ValueError("PSF is degenerate. Check motion vectors.")

    psf /= np.sum(psf)

    return psf


def smooth_motion_vectors(motion_vectors, sigma=1.0):
    """
    Smooth the motion vectors using Gaussian filtering to reduce noise.

    Parameters:
        motion_vectors (list of tuples): A list of (dx, dy) motion vectors.
        sigma (float): Standard deviation for Gaussian kernel.

    Returns:
        smoothed_vectors (list of tuples): Smoothed motion vectors.
    """
    dx = np.array([vec[0] for vec in motion_vectors])
    dy = np.array([vec[1] for vec in motion_vectors])

    smoothed_dx = gaussian_filter1d(dx, sigma)
    smoothed_dy = gaussian_filter1d(dy, sigma)

    return list(zip(smoothed_dx, smoothed_dy))


def plot_motion_path(motion_vectors):
    """
    Plot the motion path from motion vectors.

    Parameters:
        motion_vectors (list of tuples): A list of (dx, dy) motion vectors.
    """
    dx = [vec[0] for vec in motion_vectors]
    dy = [vec[1] for vec in motion_vectors]

    x_path = np.cumsum(dx)
    y_path = np.cumsum(dy)

    plt.figure(figsize=(8, 6))
    plt.plot(x_path, y_path, marker='o', linestyle='-', label='Motion Path')
    plt.xlabel('Cumulative X Displacement')
    plt.ylabel('Cumulative Y Displacement')
    plt.title('Motion Path Visualization')
    plt.legend()
    plt.grid(True)
    plt.show()


def visualize_psf(psf):
    """
    Visualize a PSF as a heatmap.
    """
    plt.imshow(psf, cmap='gray', interpolation='nearest')
    plt.colorbar()
    plt.title("Point Spread Function (PSF)")
    plt.show()
