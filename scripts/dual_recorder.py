import time
import argparse
from pathlib import Path
from picamera2 import Picamera2, Preview
from picamera2.encoders import H264Encoder
from picamera2.outputs import FfmpegOutput

CAM_A_RESOLUTION = (1920, 1080)
CAM_B_RESOLUTION = (720, 480)

CAM_A_FRAMERATE = 10
CAM_B_FRAMERATE = 100

VID_DURATION = 10 # in seconds

# initialize argument parser
parser = argparse.ArgumentParser()
parser.add_argument("-t", action='store_true', help = "Run in test mode (won't save videos)")
parser.add_argument(
    "-f", 
    type=str, 
    metavar="BASE_NAME", 
    default="camera", 
    help="Specify a base name for the video files (e.g., 'video' will save as 'video_a.mp4' and 'video_b.mp4')."
)

parser.parse_args()
args = parser.parse_args()

camera_a = Picamera2(0)
config_a = camera_a.create_video_configuration(main={"size":CAM_A_RESOLUTION})
camera_a.configure(config_a)
camera_a.start_preview(Preview.QTGL, x=100,y=300,width=400,height=300)

camera_b = Picamera2(1)
config_b = camera_b.create_video_configuration(main={"size":CAM_B_RESOLUTION})
camera_b.configure(config_b)
camera_b.start_preview(Preview.QT, x=500,y=300,width=400,height=300)

camera_a.set_controls({"FrameRate":CAM_A_FRAMERATE})

camera_b.set_controls({"FrameRate":CAM_B_FRAMERATE})

camera_a.start()
camera_b.start()

time.sleep(2)

if args.t:
    print("This is just a test and I will not save a file")
    time.sleep(VID_DURATION)
else:
    encodera = H264Encoder(1000000)
    encoderb = H264Encoder(1000000)

    output_a = FfmpegOutput(f"{args.f}_a.mp4")
    output_b = FfmpegOutput(f"{args.f}_b.mp4")

    camera_a.start_recording(encodera,output_a)
    camera_b.start_recording(encoderb,output_b)

    time.sleep(VID_DURATION)

    camera_a.stop_recording()
    camera_b.stop_recording()

    print("Files saved")
    

camera_a.stop_preview()
camera_b.stop_preview()
