import cv2 as cv
import numpy as np

def extract_motion_vectors(video_path, use_global_motion=True, center_weight=0.2):
    """
    Extract motion vectors from a video using optical flow, with optional center prioritization.

    Parameters:
        video_path (str): Path to the video file.
        use_global_motion (bool): Whether to calculate global motion vectors.
        center_weight (float): Fraction of the image dimensions to prioritize central features.

    Returns:
        motion_vectors (list of tuples): Extracted motion vectors.
    """
    cap = cv.VideoCapture(video_path)
    feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
    lk_params = dict(winSize=(50, 50), maxLevel=2,
                     criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

    ret, old_frame = cap.read()
    if not ret:
        raise ValueError("Error reading video.")

    height, width = old_frame.shape[:2]
    center_x, center_y = width // 2, height // 2
    max_dist_x = center_weight * width
    max_dist_y = center_weight * height

    old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
    corners = cv.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

    # Filter corners to prioritize central features
    filtered_corners = []
    if corners is not None:
        for corner in corners:
            x, y = corner.ravel()
            if abs(x - center_x) <= max_dist_x and abs(y - center_y) <= max_dist_y:
                filtered_corners.append([x, y])
    p0 = np.array(filtered_corners, dtype=np.float32).reshape(-1, 1, 2)

    motion_vectors = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

        if p1 is not None:
            good_new = p1[st == 1]
            good_old = p0[st == 1]

            if use_global_motion:
                mean_dx = np.mean(good_new[:, 0] - good_old[:, 0])
                mean_dy = np.mean(good_new[:, 1] - good_old[:, 1])
                motion_vectors.append((mean_dx, mean_dy))
            else:
                for new, old in zip(good_new, good_old):
                    dx = new[0] - old[0]
                    dy = new[1] - old[1]
                    motion_vectors.append((dx, dy))

        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)

    cap.release()
    return motion_vectors

def determine_kernel_size(video_path, scale=0.02, min_size=15, max_size=101):
    """
    Determine the PSF kernel size dynamically based on video resolution.

    Parameters:
        video_path (str): Path to the video file.
        scale (float): Fraction of the maximum video dimension for kernel size.
        min_size (int): Minimum allowed kernel size.
        max_size (int): Maximum allowed kernel size.

    Returns:
        psf_size (int): Dynamically determined kernel size.
    """
    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Unable to open video: {video_path}")

    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    max_dim = max(width, height)
    psf_size = int(scale * max_dim)
    psf_size = max(min_size, min(psf_size | 1, max_size))
    return psf_size

def visualize_optical_flow(video_path, center_weight=0.2, save_frame=False):
    """
    Visualize optical flow, prioritizing objects in the center of the video.

    Parameters:
        video_path (str): Path to the video file.
        center_weight (float): Fraction of the image dimensions to prioritize central features.
        save_frame (bool): Whether to save the final frame as an image.

    Returns:
        final_frame (numpy.ndarray): The last processed frame with optical flow visualization (if save_frame=True).
    """
    cap = cv.VideoCapture(video_path)
    feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
    lk_params = dict(winSize=(50, 50), maxLevel=2,
                     criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

    ret, old_frame = cap.read()
    if not ret:
        raise ValueError("Error reading video.")

    # Dimensions for center prioritization
    height, width = old_frame.shape[:2]
    center_x, center_y = width // 2, height // 2
    max_dist_x = center_weight * width
    max_dist_y = center_weight * height

    old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
    corners = cv.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

    # Filter corners to prioritize central features
    filtered_corners = []
    if corners is not None:
        for corner in corners:
            x, y = corner.ravel()
            if abs(x - center_x) <= max_dist_x and abs(y - center_y) <= max_dist_y:
                filtered_corners.append([x, y])
    p0 = np.array(filtered_corners, dtype=np.float32).reshape(-1, 1, 2)

    mask = np.zeros_like(old_frame)
    color = np.random.randint(0, 255, (100, 3))

    final_frame = None  # Store the last processed frame

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

        if p1 is not None:
            good_new = p1[st == 1]
            good_old = p0[st == 1]
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                c, d = old.ravel()
                mask = cv.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
                frame = cv.circle(frame, (int(a), int(b)), 5, color[i].tolist(), -1)
            img = cv.add(frame, mask)
            final_frame = img  # Update the final frame with visualization
            cv.imshow('Optical Flow', img)

        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)

        if cv.waitKey(30) & 0xFF == 27:  # Exit on ESC key
            break

    cap.release()
    cv.destroyAllWindows()

    if save_frame:
        return final_frame