import cv2
import numpy as np
from pprint import pprint
import matplotlib.pyplot as plt

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
    aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
    corners, ids, _ = aruco_detector.detectMarkers(gray)

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
# Extract X and Y coordinates
x_positions = np.array(positions)[:, 0]
y_positions = np.array(positions)[:, 1]

# Plotting the positions on an XY plane
plt.scatter(x_positions, y_positions, c='green', label='Marker Centers')
plt.xlabel('X Coordinates')
plt.ylabel('Y Coordinates')
plt.title('Positions of Detected ArUco Markers')
plt.legend()
plt.show()
np.save('positions.npy', positions)
# cap.release()
# cv2.destroyAllWindows()
