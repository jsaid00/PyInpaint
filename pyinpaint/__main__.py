import os
import argparse
from PIL import Image
import numpy as np
from pyinpaint.inpaint import Inpainting


def main():
    # Setting up command-line arguments
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--org_img", type=str, help="The path to the original image.")
    argparser.add_argument("--mask", type=str, help="The path to the mask image.")
    argparser.add_argument("--ps", type=int, default=7, help="Patch size.")
    argparser.add_argument("--k_boundary", type=int, default=4, help="To determine the boundary pixels. Ideally should be 4 or 8.")
    argparser.add_argument("--k_search", type=int, default=1000, help="Determines the search range, normally 500 or 1000")
    argparser.add_argument("--k_patch", type=int, default=5, help="knn value for the non-local graph, ideal values are 3,5,7")
    args = argparser.parse_args()

    # Load the original image and mask as arrays
    org_image = Image.open(args.org_img)
    mask_image = Image.open(args.mask)

    org_array = np.array(org_image)
    mask_array = np.array(mask_image)

    # Call the Inpainting function
    inpainted_img = Inpainting(org_array, mask_array, args.ps, args.k_boundary, args.k_search, args.k_patch)

    # Construct the output file path
    file_name, file_extension = os.path.splitext(args.org_img)
    output_path = f"{file_name}_inpainted{file_extension}"

    # Convert the image data to a Pillow Image object
    inpaint_result_image = Image.fromarray(inpainted_img)

    # Save the image using Pillow
    inpaint_result_image.save(output_path)


if __name__ == "__main__":
    main()
