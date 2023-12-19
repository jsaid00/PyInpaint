from pyinpaint.inpaint import Inpainting
import SimpleITK as sitk
import os
import matplotlib.image as mpimg


def inpaint_image(image_path, mask_path):
    """
    Inpaint an image using a specified mask and save the inpainted image.

    Parameters:
    - image_path (str): Path to the original image file.
    - mask_path (str): Path to the mask image file.
    """

    # Read the image and mask from the specified paths
    img = mpimg.imread(image_path)
    mask = mpimg.imread(mask_path)

    # Perform inpainting on the image using the mask
    image_inpainted = Inpainting(img, mask)

    # Construct the output file path
    file_name, file_extension = os.path.splitext(image_path)
    output_path = f"{file_name}_inpainted{file_extension}"

    # Convert the inpainted image to SimpleITK format and save it
    img_overlay_itk = sitk.GetImageFromArray(image_inpainted.astype('uint8'))
    sitk.WriteImage(img_overlay_itk, output_path)


# Specify the paths to the image and the mask
img_path = "/Users/jawhersaid/Downloads/pythonProject2/PyInpaint_V2/data/image.bmp"
mask_path = "/Users/jawhersaid/Downloads/pythonProject2/PyInpaint_V2/data/mask.bmp"

# Call the inpainting function with the image and mask paths
inpaint_image(img_path, mask_path)
