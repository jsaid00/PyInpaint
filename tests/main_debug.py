from pyinpaint.inpaint import Inpainting
import SimpleITK as sitk
import os


def inpaint_image(image_path, mask_path_):
    """
    Inpaint an image using a specified mask and save the inpainted image.

    Parameters:
    - image_path (str): Path to the original image file.
    - mask_path (str): Path to the mask image file.
    """

    # Perform inpainting on the image using the mask
    image_inpainted = Inpainting(image_path, mask_path_)

    # Construct the output file path
    file_name, file_extension = os.path.splitext(image_path)
    output_path = f"{file_name}_inpainted{file_extension}"

    # Convert the inpainted image to SimpleITK format and save it
    sitk.WriteImage(image_inpainted, output_path)


def main():
    # Set the paths relative to the current script's directory
    script_dir = os.path.dirname(__file__)  # Gets the directory where the script is located
    root_dir = os.path.dirname(script_dir)  # Navigates up to the root directory (PyInpaint)
    img_path = os.path.join(root_dir, "data/image.bmp")
    mask_path = os.path.join(root_dir, "data/mask.bmp")

    # Call the inpainting function with the image and mask paths
    inpaint_image(img_path, mask_path)


if __name__ == "__main__":
    main()
