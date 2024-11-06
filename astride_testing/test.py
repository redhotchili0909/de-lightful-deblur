from PIL import Image
import numpy as np
from astropy.io import fits
from astride import Streak
import os

img_path = 'fits_images/2.jpg'
img = Image.open(img_path).convert('L') 
image_data = np.array(img)

fits_output_dir = 'results'
os.makedirs(fits_output_dir, exist_ok=True)

file_name = os.path.basename(img_path).replace('.jpg', '.fits')
fits_file_path = os.path.join(fits_output_dir, file_name)

hdu = fits.PrimaryHDU(image_data)
hdu.writeto(fits_file_path, overwrite=True)

streak = Streak(fits_file_path)
streak.detect()

streak.write_outputs()
streak.plot_figures()

print("Streak detection completed")
