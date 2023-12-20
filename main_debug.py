import os
from pyinpaint.inpaint import Inpainting
import matplotlib.image as mpimg
from PIL import Image


def inpaint_image(image_path, mask_path_):
    """
    Inpaint an image using a specified mask and save the inpainted image.

    Parameters:
    - image_path (str): Path to the original image file.
    - mask_path (str): Path to the mask image file.
    """
    # Read the image and mask from the specified paths
    image = mpimg.imread(image_path)
    mask = mpimg.imread(mask_path_)

    # Perform inpainting on the image using the mask
    image_inpainted = Inpainting(image, mask)

    # Construct the output file path
    file_name, file_extension = os.path.splitext(image_path)
    output_path = f"{file_name}_inpainted{file_extension}"

    # Convert the image data to a Pillow Image object
    inpaint_result_image = Image.fromarray(image_inpainted)

    # Save the image using Pillow
    inpaint_result_image.save(output_path)


def main():
    # Set the paths relative to the current script's directory
    script_dir = os.path.dirname(__file__)  # Gets the directory where the script is located
    image_path = os.path.join(script_dir, "data/lincoln_grayscale.png")
    mask_path = os.path.join(script_dir, "data/lincoln_mask.png")

    # Call the inpainting function with the image and mask paths
    inpaint_image(image_path, mask_path)


if __name__ == "__main__":
    main()
