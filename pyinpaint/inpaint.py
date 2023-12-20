import numpy as np
from scipy import spatial
from tqdm import tqdm
import matplotlib.image as mpimg
from PIL import Image
from numpy.lib.stride_tricks import as_strided


def position_matrix(shape):
    """
     Generates a position matrix for the image.

     Args:
         shape (tuple): Shape of the image (height, width).

     Returns:
         numpy.ndarray: Position matrix with each row corresponding to the position of a pixel.
     """
    # Extract height and width, ignoring color channels if present.
    shape = shape[:2]
    # Create a meshgrid for each dimension in shape
    grids = np.meshgrid(*[np.arange(s) for s in shape], indexing='ij')

    # Stack and reshape the grids to form the position matrix
    return np.stack(grids, axis=-1).reshape(-1, len(shape))


def create_patches(img, patch_shape=(3, 3, 3)):
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


# Function to preprocess the image
def preprocess(img, mask_, patch_size):
    """
    Preprocesses the image for inpainting by normalizing and masking.

    Args:
        img (numpy.ndarray): The input image.
        mask_ (numpy.ndarray): Mask to indicate the area to inpaint.
        patch_size (int): Patch size.

    Returns:
        Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, tuple]: Processed position, texture, patches, and shape of the image.
    """
    # Normalize the image to a range of 0 to 1.
    img = (img - np.min(img)) / (np.max(img) - np.min(img)) + 0.01
    # Apply the mask to the image.
    img = (img.T * mask_.T).T
    # Store the shape of the image.
    shape = img.shape
    # Generate a position matrix for the image.
    position = position_matrix(shape)
    # Flatten the image for processing.
    texture = img.reshape(-1, img.shape[-1] if img.ndim == 3 else 1)
    # Normalize position and texture to a range of 0 to 1.
    position = (position - np.min(position)) / (np.max(position) - np.min(position))
    texture = (texture - np.min(texture)) / (np.max(texture) - np.min(texture))
    # Create patches of the image.
    patches = create_patches(img, (patch_size, patch_size))
    return position, texture, patches, shape


# Function for the inpainting forward pass
def forward(position, texture, patches, shape, patch_size, k_boundary, k_search, k_patch):
    """
    The inpainting forward pass, performing the actual inpainting.

    Args:
        position (numpy.ndarray): Position matrix of the image.
        texture (numpy.ndarray): Texture matrix of the image.
        patches (numpy.ndarray): Patches of the image.
        shape (tuple): Shape of the image.
        patch_size (int): Size of the patches.
        k_boundary (int): Number of nearest neighbors for boundary.
        k_search (int): Number of nearest neighbors for search.
        k_patch (int): Number of nearest neighbors for patch.

    Returns:
        numpy.ndarray: Inpainted texture matrix.
    """
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
        for i in np.where(in_A)[0]:  # Iterate only over indices where in_A is True
            # Find boundary pixels using KD-tree
            _, indices = kdt.query(position[i], k_boundary)

            # Check if current pixel is at the boundary of the inpainting area
            if not in_A[indices].all():
                dmA = np.append(dmA, i)  # Add pixel to list of newly inpainted pixels

                # Create a mask to keep track of non-zero elements in the patch
                mask_ = (~(patches[i].flatten() == 0)).astype("int")

                # Find nearest neighbors for the patch search
                _, indices = kdt.query(position[i], k_search)
                not_in_A_indices = indices[~in_A[indices]]

                # Calculate new patch values based on nearest neighbors
                new_patches = mask_.flatten() * patches[not_in_A_indices]
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
    return texture


# Main inpainting function
def Inpainting(image_array, mask_array, patch_size=7, k_boundary=4, k_search=1000, k_patch=5):
    """
    Main inpainting function. Orchestrates the inpainting process using the above functions.

    Args:
        image_array (numpy.ndarray): The input image.
        mask_array (numpy.ndarray): The mask indicating regions to inpaint.
        patch_size (int): Size of the patches used for inpainting.
        k_boundary (int): Number of nearest neighbors to consider for boundary detection.
        k_search (int): Number of nearest neighbors to search in the patch matching.
        k_patch (int): Number of nearest neighbors to consider for each patch.

    Returns:
        numpy.ndarray: The inpainted image.
    """

    # Preprocess the image and mask.
    position, texture, patches, shape = preprocess(image_array, mask_array, patch_size)

    # Perform the inpainting forward pass.
    texture = forward(position, texture, patches, shape, patch_size, k_boundary, k_search, k_patch)

    # Reshape the texture to form the inpainted image.
    inpainted_image = np.reshape(texture, shape)

    return (inpainted_image * 255).astype(np.uint8)