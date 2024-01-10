import os
import numpy as np
from pyinpaint.inpaint import Inpainting
import matplotlib.image as mpimg
from PIL import Image


def inpaint_image(image_path, mask_path):
    """
    Inpaint an image using a specified mask and save the inpainted image.

    Parameters:
    - image_path (str): Path to the original image file.
    - mask_path (str): Path to the mask image file.
    """
    # Read the image and mask using PIL
    image = Image.open(image_path)
    mask = Image.open(mask_path)

    # Convert the PIL images to NumPy arrays and ensure they are in uint8 format
    image_array = np.array(image).astype(np.uint8)
    mask_array = np.array(mask).astype(np.uint8)

    # Perform inpainting on the image using the mask
    image_inpainted = Inpainting(image_array, mask_array)

    # Ensure the output image is in uint8 format
    image_inpainted = image_inpainted.astype(np.uint8)

    # Construct the output file path
    file_name, file_extension = os.path.splitext(image_path)
    output_path = f"{file_name}_inpainted{file_extension}"

    # Convert the inpainted image data back to a Pillow Image object
    inpaint_result_image = Image.fromarray(image_inpainted)

    # Save the image using Pillow
    inpaint_result_image.save(output_path)


def main():
    # Set the paths relative to the current script's directory
    script_dir = os.path.dirname(__file__)  # Gets the directory where the script is located
    image_path = os.path.join(script_dir, "data/temp_image.bmp")
    mask_path = os.path.join(script_dir, "data/temp_mask.bmp")

    # Call the inpainting function with the image and mask paths
    inpaint_image(image_path, mask_path)


if __name__ == "__main__":
    main()
