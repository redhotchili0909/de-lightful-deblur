import argparse
from picamera2 import Picamera2, Preview
from picamera2.encoders import H264Encoder
from libcamera import controls
import time
import threading
from datetime import datetime

def config_cameras() -> tuple[Picamera2]:
    camera_a = Picamera2(camera_num=0)  # High-resolution camera
    camera_b = Picamera2(camera_num=1)  # Low-resolution camera

    # Configure Camera A for high-resolution capture
    config_a = camera_a.create_still_configuration(
        main={"size": (4608, 2592)}, raw={"format": "SBGGR10", "size": (4608, 2592)}
    )
    camera_a.configure(config_a)

    # Configure Camera B for low-resolution video capture
    config_b = camera_b.create_video_configuration(
        main={"size": (768, 432)},
        raw={"format": "SRGGB10_CSI2P", "size": (1536, 864)},
        controls={"FrameRate": 120},
    )
    camera_b.configure(config_b)

    camera_a.start()
    camera_b.start()

    time.sleep(1)

    camera_a.set_controls(
        {
            "AeEnable": False,
            "AwbEnable": False,
            "AfMode": controls.AfModeEnum.Continuous,
            "AfSpeed": controls.AfSpeedEnum.Fast,
            "AnalogueGain": 1.0,
            "ExposureTime": capture_duration_us,
        }
    )

    time.sleep(2)

    return camera_a, camera_b



def capture_images_and_video(capture_duration_us: int, output_video: str, output_image:str):
    camera_a, camera_b = config_cameras()
    encoder = H264Encoder(bitrate=10000000)

    def capture_high_res():
        """Capture a single high-resolution image."""
        start_time = datetime.now()
        print(f"[Camera A Start]: {start_time}")
        r = camera_a.capture_request(flush=True)
        r.save("main", output_image)
        end_time = datetime.now()
        print(f"[Camera A Finish]: {end_time}")
        r.release()

    def capture_low_res_video():
        """Capture a low-resolution video."""
        start_time = datetime.now()
        print(f"[Camera B Start]: {start_time}")
        camera_b.start_recording(encoder, output_video)
        time.sleep(capture_duration_us / 1e6)
        camera_b.stop_recording()
        end_time = datetime.now()
        print(f"[Camera B Finish]: {end_time}")
    
    # Threading ensures simultaneous operation
    thread_a = threading.Thread(target=capture_high_res)
    thread_b = threading.Thread(target=capture_low_res_video)

    # Start threads
    thread_a.start()
    thread_b.start()

    # Wait for both threads to complete
    thread_a.join()
    thread_b.join()

    # Cleanup
    camera_a.stop()
    camera_b.stop()
    print("Capture complete.")

if __name__ == "__main__":
    # Setup argument parser
    parser = argparse.ArgumentParser(description="Capture high-resolution images and low-resolution videos.")
    parser.add_argument(
        "--duration", 
        type=float, 
        help="Capture duration in seconds (default: 0.4 seconds)", 
        default=0.4
    )
    parser.add_argument(
        "--video_output", 
        type=str, 
        help="Output file name for the video (default: output_video)", 
        default="output_video"
    )
    parser.add_argument(
        "--image_output", 
        type=str, 
        help="Output file name for the high res image (default: high_res_image)", 
        default="high_res_image"
    )
    args = parser.parse_args()
    capture_duration_us = int(args.duration * 1e6)
    capture_images_and_video(capture_duration_us, args.video_output + ".h264", args.image_output +".jpg")