import cv2
import os
import argparse

def extract_frames(video_file, output_dir):
    """
    Extracts frames from a video file and saves them as images in the output directory.

    Args:
        video_file (str): Path to the input video file.
        output_dir (str): Path to the output directory for saving frames.
    """
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_file)

    if not cap.isOpened():
        print("Error: Could not open video file.")
        return
    
    frame_count = 0

    while True:
        ret, frame = cap.read() 
        if not ret:
            break 

        frame_file = os.path.join(output_dir, f"frame_{frame_count:04d}.jpg")
        cv2.imwrite(frame_file, frame)
        frame_count += 1

        print(f"Saved {frame_file}")

    cap.release()
    print(f"Extracted {frame_count} frames to {output_dir}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract frames from a video file.")
    parser.add_argument(
        "--video_file", type=str, help="Path to the input video file.", default="output_video.h264"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        help="Path to the output directory for frames.", 
        default="frames"
    )

    args = parser.parse_args()
    extract_frames(args.video_file, args.output_dir)