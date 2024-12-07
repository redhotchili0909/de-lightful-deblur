from PIL import Image

# Load the image
input_file = "high_res_image.jpg"
output_file = "cropped_image.jpg"

# Open the image
image = Image.open(input_file)

# Original dimensions
orig_width, orig_height = image.size

# Target dimensions
target_width, target_height = 3072, 1728

# Calculate the coordinates for the center crop
left = (orig_width - target_width) // 2
top = (orig_height - target_height) // 2
right = left + target_width
bottom = top + target_height

# Crop the image
cropped_image = image.crop((left, top, right, bottom))

# Save the cropped image
cropped_image.save(output_file)
print(f"Cropped image saved as {output_file}")
