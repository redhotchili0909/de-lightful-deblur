import cv2 as cv
import numpy as np

def extract_motion_vectors(video_path, use_global_motion=True, roi=None):
    """
    Extract motion vectors from a video using optical flow, with optional ROI prioritization.

    Parameters:
        video_path (str): Path to the video file.
        use_global_motion (bool): Whether to calculate global motion vectors.
        roi (tuple): Region of interest as (x, y, width, height). If None, use the full frame.

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
    mask = np.zeros((height, width), dtype=np.uint8)

    if roi:
        x, y, w, h = roi
        mask[y:y + h, x:x + w] = 255 
    else:
        mask[:] = 255  # Use the entire frame if no ROI is provided

    old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
    corners = cv.goodFeaturesToTrack(old_gray, mask=mask, **feature_params)
    p0 = corners.reshape(-1, 1, 2) if corners is not None else np.array([], dtype=np.float32)

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

def extract_motion_vectors_single_point(video_path, roi=None):
    """
    Extract motion vectors using optical flow, based on a single feature point.

    Parameters:
        video_path (str): Path to the video file.
        roi (list or tuple, optional): [x, y, width, height] of the region of interest.

    Returns:
        motion_vectors (list of tuples): Motion vectors (dx, dy) for the single tracked point.
    """
    cap = cv.VideoCapture(video_path)
    feature_params = dict(maxCorners=1, qualityLevel=0.3, minDistance=7, blockSize=7)
    lk_params = dict(winSize=(50, 50), maxLevel=2,
                     criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

    ret, old_frame = cap.read()
    if not ret:
        raise ValueError("Error reading video.")

    # Convert the frame to grayscale
    old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)

    # Create a mask for the ROI if specified
    mask = None
    if roi is not None:
        mask = np.zeros_like(old_gray)
        x, y, w, h = roi
        mask[y:y + h, x:x + w] = 255

    # Detect the single strongest corner in the ROI
    p0 = cv.goodFeaturesToTrack(old_gray, mask=mask, **feature_params)

    if p0 is None or len(p0) == 0:
        raise ValueError("No feature point found in the specified ROI.")

    motion_vectors = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

        if p1 is not None and len(p1) > 0:
            # Track the motion of the single feature point
            new_point = p1[0].ravel()
            old_point = p0[0].ravel()

            dx = new_point[0] - old_point[0]
            dy = new_point[1] - old_point[1]
            motion_vectors.append((dx, dy))

            # Update the old frame and point for the next iteration
            old_gray = frame_gray.copy()
            p0 = p1
        else:
            break

    cap.release()
    return motion_vectors

def visualize_optical_flow(video_path, roi=None, save_frame=False):
    """
    Visualize optical flow, prioritizing features within a specific ROI.

    Parameters:
        video_path (str): Path to the video file.
        roi (tuple): Region of interest as (x, y, width, height). If None, use the full frame.
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

    height, width = old_frame.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)

    if roi:
        # Define the mask for the region of interest (ROI)
        x, y, w, h = roi
        mask[y:y + h, x:x + w] = 255  # ROI is white, rest is black
    else:
        mask[:] = 255  # Use the entire frame if no ROI is provided

    old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
    corners = cv.goodFeaturesToTrack(old_gray, mask=mask, **feature_params)
    p0 = corners.reshape(-1, 1, 2) if corners is not None else np.array([], dtype=np.float32)

    motion_mask = np.zeros_like(old_frame)
    color = np.random.randint(0, 255, (100, 3))

    final_frame = None

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
                motion_mask = cv.line(motion_mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
                frame = cv.circle(frame, (int(a), int(b)), 5, color[i].tolist(), -1)
            img = cv.add(frame, motion_mask)
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
