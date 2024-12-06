import time
from pathlib import Path
from picamera2 import Picamera2, Preview
from picamera2.encoders import H264Encoder
from picamera2.outputs import FfmpegOutput

output_dir.mkdir(parents=True,exist_ok=True)

resolution = (1920,1080)

camera_a_framerate = 10
camera_b_framerate = 100
video_length_seconds = 10

camera_a = Picamera2(0)
config_a = camera_a.create_video_configuration(main={"size":resolution})
camera_a.configure(config_a)
camera_a.start_preview(Preview.QTGL, x=100,y=300,width=400,height=300)

camera_b = Picamera2(1)
config_b = camera_b.create_video_configuration(main={"size":resolution})
camera_b.configure(config_b)
camera_b.start_preview(Preview.QT, x=500,y=300,width=400,height=300)

camera_a.set_controls({"FrameRate":camera_a_framerate})

camera_b.set_controls({"FrameRate":camera_b_framerate})

camera_a.start()
camera_b.start()

time.sleep(2)

encodera = H264Encoder(1000000)
encoderb = H264Encoder(1000000)

output_a = FfmpegOutput('camera_a.mp4')
output_b = FfmpegOutput('camera_b.mp4')

camera_a.start_recording(encodera,output_a)
camera_b.start_recording(encoderb,output_b)

time.sleep(video_length_seconds)

camera_a.stop_recording()
camera_b.stop_recording()

camera_a.stop_preview()
camera_b.stop_preview()
