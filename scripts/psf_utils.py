import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt

def calculate_psf(motion_vectors, psf_size=21, num_samples=100):
    """
    Calculate the Point Spread Function (PSF) from motion vectors.

    Parameters:
        motion_vectors (list of tuples): A list of (dx, dy) motion vectors.
        psf_size (int): Size of the PSF kernel (must be odd, e.g., 21x21).
        num_samples (int): Number of interpolation points for the motion path.

    Returns:
        psf (numpy.ndarray): A 2D PSF kernel normalized to sum to 1.
    """
    # Extract the horizontal (dx) and vertical (dy) motion components from the vectors
    dx = [vec[0] for vec in motion_vectors]
    dy = [vec[1] for vec in motion_vectors]

    # Compute the cumulative motion path (i.e., the trajectory of motion over time)
    x_path = np.cumsum(dx)
    y_path = np.cumsum(dy)

    # Normalize the cumulative path to fit within the kernel's bounds

    # Subtract the minimum to ensure the path starts at (0, 0)
    x_path -= np.min(x_path)
    y_path -= np.min(y_path)

    # Scale the path so it fits within the PSF size
    x_path = (x_path / np.max(x_path)) * (psf_size - 1)
    y_path = (y_path / np.max(y_path)) * (psf_size - 1)

    # Interpolate the motion path to create a smoother trajectory

    # Generate a normalized time array based on the number of points in the path
    t = np.linspace(0, 1, len(x_path))

    # Generate a higher-resolution time array for smoother interpolation
    t_interp = np.linspace(0, 1, num_samples)

    # Perform cubic spline interpolation on both x and y paths
    spline_x = CubicSpline(t, x_path)
    spline_y = CubicSpline(t, y_path)

    # Get the interpolated x and y coordinates of the motion path
    x_interp = spline_x(t_interp)
    y_interp = spline_y(t_interp)

    # Initialize the PSF kernel as a 2D array filled with zeros
    psf = np.zeros((psf_size, psf_size))

    # Map the interpolated motion path to the PSF kernel
    
    # For each point in the interpolated path, increment the corresponding kernel cell
    for x, y in zip(x_interp, y_interp):
        psf[int(y), int(x)] += 1

    # Normalize the PSF so that its sum equals 1 (ensuring energy conservation)
    psf /= np.sum(psf)

    return psf

def visualize_side_by_side(global_psf, individual_psf):
    """
    Visualize two PSFs side by side for comparison.

    Parameters:
        global_psf (numpy.ndarray): PSF for global motion.
        individual_psf (numpy.ndarray): PSF for multiple motions.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(global_psf, cmap='hot', interpolation='nearest')
    axes[0].set_title('Global Motion PSF')
    axes[0].axis('off')

    axes[1].imshow(individual_psf, cmap='hot', interpolation='nearest')
    axes[1].set_title('Individual Motion PSF')
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()
