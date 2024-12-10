import cv2
import numpy as np
from pprint import pprint

# Load the video
video_path = "assets/vids/output_video.mp4"
cap = cv2.VideoCapture(video_path)
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
parameters = cv2.aruco.DetectorParameters()
positions = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = cv2.aruco.ArucoDetector(aruco_dict, parameters).detectMarkers(gray)

    if ids is not None and len(ids) == 1:
        # pixel coords of the single detected marker
        corner = corners[0][0]  # record marker corners
        top_left = corner[0]
        top_right = corner[1]
        bottom_right = corner[2]
        bottom_left = corner[3]
        center = np.mean(corner, axis=0)

        print(f"Frame: {int(cap.get(cv2.CAP_PROP_POS_FRAMES))} | "
              f"Top Left: {top_left} | Top Right: {top_right} | "
              f"Bottom Right: {bottom_right} | Bottom Left: {bottom_left} | Center: {center}")
        
        positions.append(center)

        cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        cv2.circle(frame, tuple(center.astype(int)), 5, (0, 255, 0), -1)

    cv2.imshow('Frame', frame)
    # press q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

pprint(positions)
cap.release()
cv2.destroyAllWindows()
