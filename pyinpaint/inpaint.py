import SimpleITK as sitk
import numpy as np
from numpy.lib.stride_tricks import as_strided
from scipy import spatial
from tqdm import tqdm


def create_patches(img, patch_shape=(3, 3)):
    """
    Creates patches from the input image.

    Args:
        img (numpy.ndarray): Input image, can be grayscale or color.
        patch_shape (tuple): Shape (height, width) of the patches to create.

    Returns:
        numpy.ndarray: Array of image patches.
    """

    # Convert grayscale to 3D by adding a channel dimension if necessary
    if img.ndim == 2:
        img = img[:, :, np.newaxis]

    h, w, d = img.shape  # Height, width, depth of image
    r, c = patch_shape  # Rows and columns of each patch

    # Calculate padding to center patches on pixels
    pad_height = (r - 1) // 2
    pad_width = (c - 1) // 2
    padding = [(pad_height, pad_height), (pad_width, pad_width), (0, 0)]

    # Pad the image symmetrically
    img_padded = np.pad(img, pad_width=padding, mode='symmetric')

    # Calculate strides for the as_strided function
    stride_h, stride_w, stride_d = img_padded.strides
    patches_shape = (h, w, r, c, d)
    patches_strides = (stride_h, stride_w, stride_h, stride_w, stride_d)

    # Create patches using as_strided and reshape to a 2D array
    patches = as_strided(img_padded, shape=patches_shape, strides=patches_strides)
    patches = patches.reshape(h * w, r * c * d)

    return patches


def position_matrix(shape):
    """
    Generates a position matrix for the image.

    Args:
        shape (tuple): Shape of the image (height, width).

    Returns:
        numpy.ndarray: Position matrix with each row corresponding to the position of a pixel.
    """

    # Create a meshgrid for each dimension in shape
    grids = np.meshgrid(*[np.arange(s) for s in shape], indexing='ij')

    # Stack and reshape the grids to form the position matrix
    return np.stack(grids, axis=-1).reshape(-1, len(shape))


def preprocessing(img, mask, patch_size):
    """
    Prepares the image, mask, and patch data for inpainting.

    Args:
        img (numpy.ndarray): The image to be inpainted.
        mask (numpy.ndarray): The mask indicating areas to inpaint.
        patch_size (int): Size of the patches to use.

    Returns:
        tuple: Contains prepared image, mask, and patch data.
    """

    # Normalize and mask the image
    img = (img - np.min(img)) / (np.max(img) - np.min(img)).astype("float32") + 0.01
    img = (img.T * mask.T).T
    shape = img.shape

    # Generate position and texture matrices
    position = position_matrix(shape)
    texture = img.reshape(-1, img.shape[-1] if img.ndim == 3 else 1)
    pos_min, pos_max = np.min(position), np.max(position)
    text_min, text_max = np.min(texture), np.max(texture)
    position = (position - pos_min) / (pos_max - pos_min)
    texture = (texture - text_min) / (text_max - text_min)

    # Create image patches
    patches = create_patches(img, (patch_size, patch_size))
    return img, mask, patch_size, shape, position, texture, patches


def Inpainting(image_path, mask_path, patch_size=7, k_boundary=4, k_search=1000, k_patch=5):
    """
    Performs inpainting on an image using the patch-based approach.

    Args:
        image_path (str): Path to the image file to be inpainted.
        mask_path (str): Path to the mask file, which indicates the regions to inpaint.
        patch_size (int, optional): Size of the square patch. Defaults to 7.
        k_boundary (int, optional): Number of nearest neighbors to consider for boundary determination. Defaults to 4.
        k_search (int, optional): Number of nearest neighbors to consider during the patch search phase. Defaults to 1000.
        k_patch (int, optional): Number of nearest patches to consider for calculating the new patch value. Defaults to 5.

    Returns:
        SimpleITK.Image: The inpainted image as a SimpleITK Image object.
    """

    # # Read the image and mask from the specified paths
    # img = mpimg.imread(image_path)
    # mask = mpimg.imread(mask_path)

    img_itk = sitk.ReadImage(image_path)
    mask_itk = sitk.ReadImage(mask_path)

    img = sitk.GetArrayFromImage(img_itk)[:, :, 0]
    mask = sitk.GetArrayFromImage(mask_itk)[:, :, 0]

    # Automatically determine if the mask is inverted
    if np.mean(mask) < 0.5:
        # Invert mask if the white area is more than 50% of the mask
        mask = (255 * ~mask).astype('uint8')

    # Initial setup: loading images, creating position and texture matrices, and patches

    img, mask, patch_size, shape, position, texture, patches = preprocessing(img, mask, patch_size)

    # Build a KD-tree for efficient spatial nearest neighbor search
    kdt = spatial.cKDTree(position)
    position_length = position.shape[0]

    # Initialize boolean arrays to keep track of pixels to inpaint (A) and inpainted pixels (dA)
    in_A = np.zeros(position_length, dtype=bool)
    in_dA = np.zeros(position_length, dtype=bool)

    # Identify initial inpainting areas based on the mask
    initial_A_positions = np.where(~texture.any(axis=1))[0]
    in_A[initial_A_positions] = True

    # Create a tqdm progress bar
    progress_bar = tqdm(total=in_A.sum(), desc="Inpainting Progress")

    # Inpainting loop: continues until all required pixels are inpainted
    while in_A.any():
        dmA = np.array([]).astype("int")  # Array to store newly inpainted pixels in this iteration
        for i, is_in_A in enumerate(in_A):
            if not is_in_A:
                continue  # Skip already inpainted or not required pixels

            # Find boundary pixels using KD-tree
            _, indices = kdt.query(position[i], k_boundary)

            # Check if current pixel is at the boundary of the inpainting area
            if not in_A[indices].all():
                dmA = np.append(dmA, i)  # Add pixel to list of newly inpainted pixels

                # Create a mask to keep track of non-zero elements in the patch
                mask = (~(patches[i].flatten() == 0)).astype("int")

                # Find nearest neighbors for the patch search
                _, indices = kdt.query(position[i], k_search)
                not_in_A_indices = indices[~in_A[indices]]

                # Calculate new patch values based on nearest neighbors
                new_patches = mask.flatten() * patches[not_in_A_indices]
                kdt_ = spatial.cKDTree(new_patches)
                _, indices = kdt_.query(patches[i].flatten(), k_patch)
                ids = not_in_A_indices[indices]

                # Update the texture matrix with the average of best matching patches
                texture[i] = texture[ids].mean(axis=0)

                # Update the progress bar
                progress_bar.update(1)

        # After each iteration, update patches with new texture values
        patches = create_patches(np.reshape(texture, shape), (patch_size, patch_size))

        # Update boolean arrays to track inpainted areas
        in_A[dmA] = False  # Mark newly inpainted pixels as not in A
        in_dA[dmA] = True  # Mark newly inpainted pixels as in dA

    progress_bar.close()
    # reshape the inpainted image
    image_inpainted = np.reshape(texture, shape)

    # Rescale the image to 0-255 and convert to uint8
    image_inpainted = (image_inpainted * 255).astype('uint8')

    return sitk.GetImageFromArray(image_inpainted.astype('uint8'))  # Convert the inpainted image to SimpleITK format
