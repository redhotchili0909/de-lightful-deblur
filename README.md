# De-Lightful Deblur

This project implements a motion deblurring pipeline based on techniques outlined in [*Motion Deblurring Using Motion Vectors*](https://ieeexplore.ieee.org/abstract/document/1288519). The pipeline processes video frames to extract motion paths, estimate a Point Spread Function (PSF), and restore motion-blurred images using Richardson-Lucy and Wiener deconvolution methods.

## Table of Contents

1. [Introduction](#introduction)
2. [Features](#features)
3. [Pipeline Overview](#pipeline-overview)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Results](#results)
7. [References](#references)

## Introduction

Motion blur occurs when a camera or subject moves during exposure, resulting in smeared or distorted images. This project implements a motion deblurring method that leverages motion vectors extracted from video sequences to estimate the PSF, which is then used to restore sharp images.

### Referenced Paper

The methods are inspired by *Motion Deblurring Using Motion Vectors*, which provides a comprehensive approach to reconstructing images degraded by motion blur. Our implementation adapts and extends the paper's principles to enable practical, automated deblurring.

## Features

- Extracts motion paths from video using **optical flow**.
- Dynamically estimates the kernel size for PSF calculation.
- Calculates the PSF using motion vector interpolation.
- Implements two deblurring methods:
  - Richardson-Lucy deconvolution.
  - Wiener deconvolution.
- Supports user-defined regions of interest (ROI) for localized motion analysis.
- Provides visualization of:
  - Optical flow paths.
  - Smoothed motion vector paths.
  - Calculated PSF heatmap.

## Pipeline Overview

1. **ROI Selection**: Manually select a region of interest to prioritize specific motion areas.
2. **Optical Flow Visualization**: Visualize motion paths extracted using optical flow.
3. **Motion Vector Extraction**: Calculate individual or global motion paths.
4. **PSF Estimation**: Generate the PSF based on extracted motion vectors.
5. **Deblurring**: Restore the blurred image using the Richardson-Lucy and Wiener methods.

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/your-repo/de-lightful-deblur.git
   cd de-lightful-deblur
   ```

2. Set up a Python virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # For Windows, use venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Install OpenCV with GUI support (if needed):
   ```bash
   pip install opencv-python-headless
   ```

## Usage

1. **Run the pipeline**:
   ```bash
   python scripts/optical_flow.py
   ```

2. **Select ROI**:
   - When prompted, manually select the region of interest in the first video frame.

3. **Input Paths**:
   - Modify `video_path` and `image_path` variables in `optical_flow.py` to point to your assets.

4. **Output**:
   - The results, including optical flow visualization, motion path plots, PSF heatmaps, and deblurred images, will be saved in the `assets/output/` directory.

## Results

The pipeline generates the following outputs:
- **Optical Flow Visualization**: Tracks motion paths from the selected ROI.
- **PSF Heatmap**: Represents the motion path used for deblurring.
- **Deblurred Images**:
  - Richardson-Lucy deblurred image.
  - Wiener deblurred image.

## References

- *Motion Deblurring Using Motion Vectors*, [Author(s)], [Journal/Conference], [Year].  
- OpenCV documentation: https://docs.opencv.org  
- scikit-image documentation: https://scikit-image.org/docs/stable/  