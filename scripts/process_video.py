import cv2 as cv
import numpy as np

def extract_motion_vectors(video_path, use_global_motion=True):
    """
    Extract motion vectors from a video using optical flow.

    Parameters:
        video_path (str): Path to the video file.
        use_global_motion (bool): Whether to calculate global motion vectors.

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

    old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
    p0 = cv.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

    motion_vectors = []
    individual_vectors = []

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
                    individual_vectors.append((dx, dy))

        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)

    cap.release()
    return motion_vectors if use_global_motion else individual_vectors

def get_video_resolution(video_path):
    """
    Extract the resolution of the video.

    Parameters:
        video_path (str): Path to the video file.

    Returns:
        width (int): Width of the video frames.
        height (int): Height of the video frames.
    """
    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Unable to open video: {video_path}")

    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    return width, height

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
    width, height = get_video_resolution(video_path)
    max_dim = max(width, height)

    # Calculate kernel size based on scale
    psf_size = int(scale * max_dim)

    # Ensure kernel size is odd and within bounds
    psf_size = max(min_size, min(psf_size | 1, max_size))
    return psf_size

def visualize_optical_flow(video_path):
    """
    Display the video with optical flow visualized in real-time, similar to the original provided implementation.
    """
    cap = cv.VideoCapture(video_path)

    feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
    lk_params = dict(winSize=(50, 50), maxLevel=2,
                     criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

    ret, old_frame = cap.read()
    if not ret:
        raise ValueError("Error reading video.")

    old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
    p0 = cv.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

    # Create a mask image for drawing purposes
    mask = np.zeros_like(old_frame)

    # Generate random colors for the tracks
    color = np.random.randint(0, 255, (100, 3))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # Calculate optical flow
        p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

        if p1 is not None:
            # Select good points
            good_new = p1[st == 1]
            good_old = p0[st == 1]

            # Draw the tracks
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                c, d = old.ravel()
                mask = cv.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
                frame = cv.circle(frame, (int(a), int(b)), 5, color[i].tolist(), -1)

            img = cv.add(frame, mask)

            cv.imshow('Optical Flow', img)  

        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)

        if cv.waitKey(30) & 0xFF == 27:  # Exit on ESC key
            break

    cap.release()