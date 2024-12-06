from process_video import *
from psf_utils import calculate_psf, visualize_side_by_side

# Main processing and visualization
video_path = 'assets/vids/test.mov'

# Dynamically determine kernel size
psf_size = determine_kernel_size(video_path, scale=0.02, min_size=15, max_size=101)

# Calculate PSFs for global and individual motion
global_motion_vectors = extract_motion_vectors(video_path, use_global_motion=True)
global_psf = calculate_psf(global_motion_vectors, psf_size=psf_size, num_samples=100)

individual_motion_vectors = extract_motion_vectors(video_path, use_global_motion=False)
individual_psf = calculate_psf(individual_motion_vectors, psf_size=psf_size, num_samples=100)

# Visualize optical flow in the video
visualize_optical_flow(video_path)

# Optionally visualize PSFs side by side
visualize_side_by_side(global_psf, individual_psf)

print(f"Video resolution: {get_video_resolution(video_path)}")
print(f"Dynamic kernel size: {psf_size}")
