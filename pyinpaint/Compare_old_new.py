#### !!!!Remarques!!!: - make the PyInpaint directory as source root directory for the code to run
####                   - Barbara image comparison is showing some difference at first run but if you run again is shows 0
####                   - there is a problem with the grayscale images inpainting that should be fixed
####                   - fix the notebook and make sure the code is running as a command line


import os
from matplotlib import pyplot as plt, image as mpimg
from pyinpaint.inpaint import *


def modify_path(path):
    file_name, file_extension = os.path.splitext(path)
    return file_name + "_inpainted" + file_extension


def compare_images(img_path1, img_path2, image_name):
    # read images
    image_array1 = mpimg.imread(img_path1)
    image_array2 = mpimg.imread(img_path2)

    # Calculate pixel-wise absolute difference
    pixel_difference = np.abs(image_array1 - image_array2)

    # Count the number of pixels that are different
    different_pixels_count = np.sum(np.any(pixel_difference != 0, axis=-1))

    # Print the number of different pixels
    print(f"Number of different pixels for {image_name}: {different_pixels_count}")

    # ### DEBUG:### Plot the difference image ######
    # plt.imshow(pixel_difference[:, :, 0])
    # plt.title("Pixel Difference")
    # plt.axis('off')
    # plt.show()


def main():
    # List of image paths
    image_paths = [
        ("/Users/jawhersaid/Downloads/pythonProject3/PyInpaint/data/lincoln.png",
         "/Users/jawhersaid/Downloads/pythonProject3/PyInpaint/data/lincoln_mask.png", "lincoln"),
        ("/Users/jawhersaid/Downloads/pythonProject3/PyInpaint/data/barbara.jpg",
         "/Users/jawhersaid/Downloads/pythonProject3/PyInpaint/data/barbara_mask.png", "barbara"),
        ("/Users/jawhersaid/Downloads/pythonProject3/PyInpaint/data/fly.png",
         "/Users/jawhersaid/Downloads/pythonProject3/PyInpaint/data/fly_mask.png", "fly")
    ]

    for img_path, mask_path, image_name in image_paths:
        # Read the original image and mask from their file paths
        img = mpimg.imread(img_path)
        mask = mpimg.imread(mask_path)

        # Create Inpaint object and generate inpainted image
        inpainted_img_array = Inpainting(img, mask, show_progress=True)
        path_to_output = modify_path(img_path)
        plt.imsave(path_to_output, inpainted_img_array)

        # Compare images and print the number of different pixels
        old_image_path = modify_path(img_path).replace("_inpainted", "_inpainted_old")
        compare_images(old_image_path, path_to_output, image_name)


# Example usage
if __name__ == '__main__':
    main()
