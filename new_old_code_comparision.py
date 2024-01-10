from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


def compare_images(img_path1, img_path2):
    # Read the images in grayscale mode
    image_pil1 = Image.open(img_path1)
    image_pil2 = Image.open(img_path2)

    # Convert the PIL images to NumPy arrays
    image_array1 = np.array(image_pil1)
    image_array2 = np.array(image_pil2)

    # Calculate pixel-wise absolute difference
    pixel_difference = np.abs(image_array1.astype(int) - image_array2.astype(int))

    # Display the results
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(image_array1, cmap='gray')
    ax[0].set_title('Old Image')
    ax[0].axis('off')

    ax[1].imshow(image_array2, cmap='gray')
    ax[1].set_title('New Image')
    ax[1].axis('off')

    ax[2].imshow(pixel_difference, cmap='gray')
    ax[2].set_title('Pixel Difference')
    ax[2].axis('off')

    plt.tight_layout()
    plt.show()


def main():
    # Paths to the images
    old_image_path = 'data/lincoln_inpainted_old.png'
    new_image_path = 'data/lincoln_inpainted_new.png'

    # Compare images visually
    compare_images(old_image_path, new_image_path)


if __name__ == '__main__':
    main()
