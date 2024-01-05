from PIL import Image, ImageChops
import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import structural_similarity as ssim
import cv2


def compare_images(img_path1, img_path2):
    # Read the images
    image_pil1 = Image.open(img_path1)
    image_pil2 = Image.open(img_path2)

    # Find the difference in images pixel by pixel using ImageChops
    difference_pil = ImageChops.difference(image_pil1, image_pil2)

    # Convert the PIL images to NumPy arrays for display
    image_array1 = np.array(image_pil1).astype(np.float32) / 255.0
    image_array2 = np.array(image_pil2).astype(np.float32) / 255.0
    difference_array = np.array(difference_pil).astype(np.float32) / 255.0

    # Calculate the intensity of the differences
    intensity = np.sqrt(np.sum(np.square(difference_array), axis=-1))
    if intensity.max() > 0:
        intensity /= intensity.max()

    # Define colormap
    colormap = plt.cm.hot

    # Figure setup with shared axes
    fig, ax = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)

    # Show the original images and the difference image
    ax[0].imshow(image_array1)
    ax[0].set_title('Old mage')
    ax[0].axis('off')

    ax[1].imshow(image_array2)
    ax[1].set_title('New Image')
    ax[1].axis('off')

    im = ax[2].imshow(intensity, cmap=colormap)
    ax[2].set_title('Intensity of Difference')
    ax[2].axis('off')

    ######################### DEBUG ##################################
    # # Identify differing pixels
    # differing_pixels = np.argwhere(intensity > 0.1)  # Threshold for significant differences
    #
    # # Overlay pixel values on the difference image
    # for (i, j) in differing_pixels:
    #     pixel_value1 = image_array1[i, j]
    #     pixel_value2 = image_array2[i, j]
    #     text = f"{pixel_value1}\n{pixel_value2}"
    #     ax[2].text(j, i, text, ha='center', va='center', color='green', fontsize=6)
    ###################################################################

    # Create a colorbar
    plt.colorbar(im, ax=ax[2], fraction=0.046, pad=0.04)

    # Adjust layout
    plt.tight_layout()
    plt.show()


def mse(imageA, imageB):
    """Calculate the Mean Squared Error between two images."""
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err


def psnr(imageA, imageB):
    """Calculate the Peak Signal to Noise Ratio between two images."""
    mse_value = mse(imageA, imageB)
    if mse_value == 0:
        return float('inf')
    max_pixel = 255.0
    return 20 * np.log10(max_pixel / np.sqrt(mse_value))


def calculate_metrics(original_img_path, img_path1, img_path2):
    # Load the original, first, and second images
    original_image = cv2.imread(original_img_path, cv2.IMREAD_COLOR)
    image1 = cv2.imread(img_path1, cv2.IMREAD_COLOR)
    image2 = cv2.imread(img_path2, cv2.IMREAD_COLOR)

    # Convert images to grayscale for SSIM
    original_gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    image1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    image2_gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # Compute MSE and SSIM between the original and the first image
    mse1 = mse(original_gray, image1_gray)
    ssim1 = ssim(original_gray, image1_gray)
    psnr1 = psnr(original_gray, image1_gray)

    # Compute MSE and SSIM between the original and the second image
    mse2 = mse(original_gray, image2_gray)
    ssim2 = ssim(original_gray, image2_gray)
    psnr2 = psnr(original_gray, image2_gray)

    return mse1, ssim1, psnr1, mse2, ssim2, psnr2


def main():
    # Paths to the images
    original_image_path = 'data/lincoln.png'
    old_image_path = 'data/lincoln_inpainted_old.png'
    new_image_path = 'data/lincoln_inpainted_new.png'

    # Compare images visually
    compare_images(old_image_path, new_image_path)

    # Calculate metrics
    metrics = calculate_metrics(original_image_path, old_image_path, new_image_path)
    print(f"Metrics for Old Image: MSE = {metrics[0]}, SSIM = {metrics[1]}, PSNR = {metrics[2]}")
    print(f"Metrics for New Image: MSE = {metrics[3]}, SSIM = {metrics[4]}, PSNR = {metrics[5]}")


if __name__ == '__main__':
    main()
