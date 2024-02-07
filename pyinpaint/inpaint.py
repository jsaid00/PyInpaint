import numpy as np
from numpy.lib.stride_tricks import as_strided
from scipy import spatial
from tqdm import tqdm


def features_matrix(img):
    """
    Converts an image to a feature matrix (N by C), where N is the number of pixels and C is the number of channels.

    Parameters:
    img (numpy.ndarray): The input image, either grayscale (2D) or color (3D).

    Returns:
    numpy.ndarray: A 2D feature matrix.
    """
    if img.ndim == 3:
        # Handle color images (3D)
        l, w, c = img.shape
        fmat = np.ones((l * w, c))
        for i in range(c):
            # Flatten each channel and stack them into the feature matrix
            fmat[:, i] = np.reshape(img[:, :, i], (l * w,))
        return fmat
    elif img.ndim == 2:
        # Handle grayscale images (2D)
        l, w = img.shape
        return img.reshape((l * w, 1))
    else:
        # Raise an error if the image is neither 2D nor 3D
        raise ValueError("Input image must be either 2D or 3D.")


def position_matrix(shape):
    """
    Generates a normalized position matrix for the image. The origin (0,0) is in the top-left corner,
    with x increasing to the right and y decreasing downwards. Positions are normalized by the
    maximum of the image's height and width.

    Args:
    shape (tuple): Shape of the image (height, width).

    Returns:
    numpy.ndarray: Normalized position matrix.
    """
    height, width = shape[:2]
    normalization_factor = max(height, width)

    # Generate x and y coordinates and normalize them
    x_coords = np.tile(np.arange(width) / normalization_factor, (height, 1))
    y_coords = np.tile(np.arange(height, 0, -1).reshape(-1, 1) / normalization_factor, (1, width))

    # Combine x and y coordinates into a 2D position matrix
    position_mat = np.dstack((x_coords, y_coords)).reshape(-1, 2)

    return position_mat


def create_patches(img, patch_shape=(3, 3, 3)):
    """
    Creates patches from the input image.

    Args:
    img (numpy.ndarray): Input image, can be grayscale or color.
    patch_shape (tuple): Shape (height, width) of the patches to create.

    Returns:
    numpy.ndarray: Array of image patches.
    """
    # Ensure the image is 3D (add a channel dimension to grayscale images)
    if img.ndim == 2:
        img = img[:, :, np.newaxis]

    h, w, d = img.shape  # Image dimensions
    r, c = patch_shape  # Patch dimensions

    # Calculate padding to center patches on pixels
    pad_height = (r - 1) // 2
    pad_width = (c - 1) // 2
    padding = [(pad_height, pad_height), (pad_width, pad_width), (0, 0)]

    # Pad the image symmetrically
    img_padded = np.pad(img, pad_width=padding, mode='symmetric')

    # Create patches using as_strided
    stride_h, stride_w, stride_d = img_padded.strides
    patches_shape = (h, w, r, c, d)
    patches_strides = (stride_h, stride_w, stride_h, stride_w, stride_d)
    patches = as_strided(img_padded, shape=patches_shape, strides=patches_strides)
    patches = patches.reshape(h * w, r * c * d)

    return patches


def preprocess(org_img, mask, patch_size):
    """
    Preprocesses the input image and mask for the inpainting process. This includes reading
    the image and mask, normalizing the image, applying the mask, and generating necessary
    matrices and patches for inpainting.

    Args:
    org_img (str): File path to the original image.
    mask (str): File path to the mask image.
    ps (int): Patch size for creating patches from the image.

    Returns:
    tuple: A tuple containing the shape of the image, the normalized position matrix,
           the normalized texture matrix, and the image patches.
    """

    # Normalize the image: Scale pixel values to range [0.01, 1]
    img = (org_img - np.min(org_img)) / (np.max(org_img) - np.min(org_img)).astype("float32") + 0.01

    # Apply the mask to the image
    img = (img.T * mask.T).T

    # Store the shape of the image
    _shape = img.shape

    # Generate the position matrix: A matrix representing the normalized coordinates of each pixel in the image
    position = position_matrix(_shape)

    # Generate the texture matrix: A matrix representing the pixel values
    texture = features_matrix(img)

    # Normalize the position and texture matrices to have values between 0 and 1
    _position = (position - np.min(position)) / (np.max(position) - np.min(position))
    _texture = (texture - np.min(texture)) / (np.max(texture) - np.min(texture))

    # Create patches from the image
    _patches = create_patches(img, (patch_size, patch_size))

    return _shape, _position, _texture, _patches


def Inpainting(org_img, mask, patch_size=7, k_boundary=4, k_search=1000, k_patch=5, show_progress=False):
    """
    Performs inpainting on an image using a texture synthesis approach.

    Args:
    org_img (str): File path to the original image.
    mask (str): File path to the mask image.
    patch_size (int): The size of the patches used for inpainting.
    k_boundary (int): The number of nearest neighbors to consider for boundary pixels.
    k_search (int): The number of nearest neighbors to consider during patch search.
    k_patch (int): The number of nearest neighbors to consider for each patch.
    show_progress (bool): If True, displays a progress bar.

    Returns:
    numpy.ndarray: The inpainted image.
    """
    # Preprocess the image and mask, and prepare the required data
    _shape, _position, _texture, _patches = preprocess(org_img, mask, patch_size)

    # Create a KD-tree for efficient nearest neighbor search in the position matrix
    kdt = spatial.cKDTree(_position)

    # Identify pixels to be inpainted (A) and already inpainted or original pixels (dA)
    dA = np.where(_texture.any(axis=1))[0]
    A = np.where(~_texture.any(axis=1))[0]

    # Initialize the progress bar if required
    if show_progress:
        pbar = tqdm(desc=f"# of pixels to be inpainted are {A.size}", total=A.size,
                    bar_format='{l_bar}{bar}|{n_fmt}/{total_fmt} - Elapsed: {elapsed}')

    # Loop through until all pixels in A are inpainted
    while A.size >= 1:
        dmA = []  # To store indices of newly inpainted pixels in each iteration

        # Iterate over each pixel to be inpainted
        for i in A:
            # Query the KD-tree for the nearest boundary pixels
            _, indices = kdt.query(_position[i], k_boundary)

            # Check if the pixel is adjacent to already inpainted pixels
            if not np.isin(indices, A).all():
                dmA.append(i)

                # Construct a mask to identify non-zero patch elements
                mask = (_patches[i].flatten() != 0).astype(int)

                # Search for similar patches in the texture
                _, indices = kdt.query(_position[i], k_search)
                part_of_dA = indices[~np.isin(indices, A)]
                new_patches = mask.flatten() * _patches[part_of_dA]

                # Create a KD-tree for the new patches and query for similar patches
                kdt_ = spatial.cKDTree(new_patches)
                _, indices = kdt_.query(_patches[i].flatten(), k_patch)

                # Update the texture of the pixel with the mean texture of similar patches
                ids = part_of_dA[indices]
                _texture[i] = _texture[ids].mean(axis=0)

        # Update the patches after each iteration
        _patches = create_patches(np.reshape(_texture, _shape), (patch_size, patch_size))

        # Update the lists of already inpainted pixels and remaining pixels
        dA = np.concatenate((dA, np.array(dmA)))
        A = np.setdiff1d(A, dmA)

        # Update the progress bar if enabled
        if show_progress:
            pbar.update(len(dmA))

    # Close the progress bar if enabled
    if show_progress:
        pbar.close()

    # Return the inpainted image, reshaped to its original dimensions
    inpainted_image = (np.reshape(_texture, _shape) * 255).astype(np.uint8)

    return inpainted_image
