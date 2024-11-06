# De-Lightful-Deblur

A Python tool for detecting light streaks in images to be used for deblurring imgaes. This repository provides both a GUI application for manual selection and processing of images, as well as a script for batch processing images in a folder.

## Table of Contents

- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
  - [Running the GUI Application](#running-the-gui-application)
  - [Batch Processing Images](#batch-processing-images)
- [License](#license)

## Features

- **Streak Detection**: Detects bright, thick light streaks in images using image processing techniques.
- **Manual Selection GUI**: Allows users to manually select regions of interest in an image for processing.
- **Batch Processing**: Automatically processes all images in a specified folder.
- **Result Visualization**: Outputs images with detected streaks outlined and numbered, along with zoomed-in images of each streak.
- **Adjustable Parameters**: Users can adjust detection parameters like contrast enhancement, threshold values, and contour area.

## Prerequisites

- Python 3.12 or higher
- [Poetry](https://python-poetry.org/docs/#installation) package manager

## Installation

Follow these steps to set up the project on your local machine:

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/de-lightful-deblur.git
cd de-lightful-deblur
```

### 2. Install Poetry

If you haven't installed Poetry yet, you can do so by following the [official installation guide](https://python-poetry.org/docs/#installation).

### 3. Install Dependencies

Use Poetry to install all the project dependencies:

```bash
poetry install
```

This command will create a virtual environment and install all the packages listed in `pyproject.toml`.

### 4. Activate the Virtual Environment

To activate the virtual environment created by Poetry, run:

```bash
poetry shell
```

## Usage

### Running the GUI Application

The GUI application allows you to manually select regions in an image and run streak detection on them.

#### Steps:

1. **Launch the Application**

   ```bash
   python gui.py
   ```

2. **Load an Image**

   - Click on the **"Load Image"** button.
   - Select the image file you want to process (supported formats: `.png`, `.jpg`, `.jpeg`).

3. **Select a Region**

   - Click and drag on the image to draw a rectangle around the region you want to process.

4. **Run Streak Detection**

   - Click on the **"Run Streak Detection"** button.
   - The selected region will be processed, and the results will be displayed on the image.
   - A message indicating that the image has been saved will appear above the image display.

5. **Reset Selection**

   - To revert back to the original image or make a new selection, click on the **"Go Back to Selection"** button.

### Batch Processing Images

You can process all images in a specified folder using the batch processing script.

#### Steps:

1. **Place Images in the Input Folder**

   - By default, the script looks for images in the `assets` folder. Place your images there or specify a different folder.

2. **Run the Batch Processing Script**

   ```bash
   python streak_detect.py
   ```

   - The script will process all images in the input folder and save the results in the `auto_select` output folder.

3. **Adjusting Parameters (Optional)**

   - You can adjust detection parameters by modifying the `process_all_images` function in `streak_detect.py`.

   ```python
   process_all_images(
       input_folder="assets",
       output_folder="auto_select",
       clip_limit=2.0,
       threshold_value=130,
       min_contour_area=80,
       min_brightness=150
   )
   ```

## Dependencies

The project uses the following Python packages:

- **OpenCV-Python**: Computer vision library for image processing.
- **NumPy**: Library for numerical computations.
- **PyQt5**: GUI toolkit for the GUI application.

All dependencies are listed in the `pyproject.toml` file and will be installed via Poetry.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.