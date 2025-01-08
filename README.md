# De-Lightful Deblur

This project implements a motion deblurring pipeline using techniques inspired by [*Motion Deblurring Using Motion Vectors*](https://ieeexplore.ieee.org/abstract/document/1288519). It uses dual-camera setups, motion vector extraction, PSF estimation, and deblurring algorithms (Richardson-Lucy and Wiener) to restore motion-blurred images. This implementation is designed for use with Raspberry Pi hardware and Python-based tools.

## Table of Contents

1. [Introduction](#introduction)
2. [Pipeline Overview](#pipeline-overview)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Code Details](#code-details)
6. [Results](#results)
8. [References](#references)


## Introduction

Motion blur occurs when a camera or subject moves during exposure, resulting in smeared images. *De-Lightful Deblur* leverages motion estimation techniques like optical flow and fiducial marker tracking to estimate motion paths and compute Point Spread Functions (PSFs), which are then used to reverse blur effects.

The project is implemented using Python, OpenCV, and Raspberry Pi, focusing on computational efficiency and accuracy for motion deblurring tasks.



## Pipeline Overview

1. **Capture and Synchronization**:
   - Simultaneous image and video capture using a Raspberry Pi dual-camera setup.
2. **Motion Estimation**:
   - Extract motion vectors using optical flow or fiducial marker tracking.
3. **PSF Generation**:
   - Compute PSF dynamically from motion paths with optional smoothing and interpolation.
4. **Deblurring**:
   - Apply deblurring algorithms to restore sharpness to blurred images.
5. **Results Visualization**:
   - Save visualizations for motion paths, PSF heatmaps, and deblurred images.


## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/de-lightful-deblur.git
   cd de-lightful-deblur
   ```

2. Set up a Python virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Install OpenCV with GUI support:
   ```bash
   pip install opencv-python
   ```

## Usage

### Image and Video Capture
1. **Run the capture script**:
   ```bash
   python scripts/capture_images_and_video.py --duration 0.4 --video_output output_video --image_output high_res_image
   ```

2. **Extract frames from the video**:
   ```bash
   python scripts/extract_frames.py --video_file output_video.h264 --output_dir frames
   ```

3. **Crop the high-resolution image to match video FOV**:
   ```bash
   python scripts/crop_center.py input_image.jpg cropped_image.jpg --width 3072 --height 1728
   ```

### Deblurring Pipeline
1. **Run the main pipeline (`optical_flow.py`)**:
   ```bash
   python scripts/optical_flow.py
   ```

2. **Provide ROI manually when prompted** or set it in the code for automation.


## Code Details

### Key Scripts
#### Raspberry Pi Scripts
- **`trigger.py`**:
  Captures synchronized high-resolution images and low-resolution videos using dual cameras.
- **`extract_frames.py`**:
  Extracts frames from video for further processing.
- **`crop.py`**:
  Crops high-resolution images to match the low-resolution video’s field of view.
#### Image Processing Scripts
- **`optical_flow.py`**:
  Main pipeline script for optical flow analysis, PSF estimation, and image deblurring.
- **`psf_utils.py`**:
  Includes functions for PSF calculation and visualization.
- **`process_video.py`**:
  Handles optical flow computation and motion vector extraction.


## Results

The pipeline generates the following outputs:
- **Optical Flow Visualizations**: Motion paths extracted from video frames.
- **PSF Heatmaps**: Represents the motion used in deblurring.
- **Deblurred Images**:
  - Richardson-Lucy deblurred result.
  - Wiener deblurred result.

Sample outputs are saved in the `assets/output/` directory with dynamically generated filenames.


## References

1. *Motion Deblurring Using Motion Vectors*, [IEEE](https://ieeexplore.ieee.org/abstract/document/1288519).
2. OpenCV documentation: https://docs.opencv.org.
3. scikit-image documentation: https://scikit-image.org/docs/stable/.