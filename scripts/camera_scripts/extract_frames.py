import cv2
import os

# Input video file
video_file = "output_video.h264"
# Output directory for frames
output_dir = "frames"

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Open the video file
cap = cv2.VideoCapture(video_file)

if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

frame_count = 0

# Read frames from the video
while True:
    ret, frame = cap.read()  # Read the next frame
    if not ret:
        break  # Exit the loop if no more frames are available

    # Save the frame as an image
    frame_file = os.path.join(output_dir, f"frame_{frame_count:04d}.jpg")
    cv2.imwrite(frame_file, frame)
    frame_count += 1

    print(f"Saved {frame_file}")

cap.release()
print(f"Extracted {frame_count} frames to {output_dir}.")
