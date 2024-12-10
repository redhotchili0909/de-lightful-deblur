import argparse
from PIL import Image

def crop_center(input_file, output_file, target_width, target_height):
    """
    Crops an image to the specified dimensions from the center.

    Args:
        input_file (str): Path to the input image file.
        output_file (str): Path to save the cropped image.
        target_width (int): Target width for the cropped image.
        target_height (int): Target height for the cropped image.
    """
    image = Image.open(input_file)
    orig_width, orig_height = image.size

    left = (orig_width - target_width) // 2
    top = (orig_height - target_height) // 2
    right = left + target_width
    bottom = top + target_height

    cropped_image = image.crop((left, top, right, bottom))

    cropped_image.save(output_file)
    print(f"Cropped image saved as {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crop an image to specified dimensions from the center.")

    parser.add_argument(
        "input_file", 
        type=str, 
        help="Path to the input image file."
    )
    parser.add_argument(
        "output_file", 
        type=str, 
        help="Path to save the cropped image.",
        nargs="?", 
        default="cropped_image.jpg"
    )
    parser.add_argument(
        "--width", 
        type=int, 
        help="Target width for the cropped image.", 
        default=3072
    )
    parser.add_argument(
        "--height", 
        type=int, 
        help="Target height for the cropped image.", 
        default=1728
    )
    
    args = parser.parse_args()
    crop_center(args.input_file, args.output_file, args.width, args.height)