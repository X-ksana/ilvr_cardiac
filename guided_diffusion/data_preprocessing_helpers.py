import numpy as np
import torch
from typing import Tuple, Optional
from skimage.transform import resize
# --- Normalisation Helper ---

def normalise_to_model_range(image_array: np.ndarray) -> np.ndarray:
    """
    Normalises an array (npy from nnU-Net) to the [-1, 1] range.
    This involves robustly clipping outliers before rescaling.
    Args:
        image_array in numpy
    Returns:
        normalised array
    """
    # Use float
    image_array = image_array.astype(np.float32)
    # Clip 
    upper_percentile = np.percentile(image_array,99.5)
    lower_percentile = np.percentile(image_array,0.05)
    clipped_image = np.clip(image_array,lower_percentile,upper_percentile)

    # Scale to [0,1]
    max_val = np.max(clipped_image)
    min_val = np.min(clipped_image)
    if max_val - min_val > 1e-6: # ensure div nonzero
        scaled_image = (clipped_image - min_val)/(max_val - min_val)
    else:
        scaled_image = np.zeros_like(clipped_image) # handle constant image
    # Shift to [-1,1]
    final_image = (scaled_image*2.0)-1.0
    return final_image


# Mask one-hot encoder helper
def one_hot_encode_mask(
        mask_array:np.ndarray,
        num_classes:int,
        drop_background:bool = True
        ) -> torch.Tensor:
    """
    Convert mask array to one hot encoder
    Drop background
    Args:
        mask_array: A 2D NumPy array (H, W) with integer class labels.
        num_classes: The total number of classes in the mask.
        drop_background: If True, the channel for class 0 is removed.

    Returns:
        A one-hot encoded PyTorch tensor of shape (C, H, W).
    """
    mask_array = mask_array.astype(np.int64)
    mask_array[mask_array < 0] = 0  # Map negative labels to background
    
    #h, w = mask_array.shape
    #one_hot = np.zeros((h, w, num_classes), dtype=np.float32)

    #for i in range(num_classes):
    #    one_hot[:, :, i] = (mask_array == i)
    one_hot = (np.arange(num_classes) == mask_array[...,None]).astype(np.float32)

    # Transpose from (H, W, C) to (C, H, W) for PyTorch
    one_hot_tensor = torch.from_numpy(one_hot).permute(2, 0, 1)

    # If drop background
    if drop_background:
        return one_hot_tensor[1:]
    else: 
        return one_hot_tensor

# Core data loading function

def load_and_process_npy_pair(
        image_slice:np.ndarray,
        mask_slice:np.ndarray,
        num_mask_classes:int,
        drop_background:bool=True,
        target_size: int = None  
) -> Tuple[torch.Tensor, int]:
    """
    Processes and combines a 2D image slice and a 2D mask slice.

    This function coordinates the normalization of the image and the
    one-hot encoding of the mask, returning a single concatenated tensor.

    Args:
        image_slice: 2D NumPy array for the image.
        mask_slice: 2D NumPy array for the mask.
        num_mask_classes: Total number of classes for one-hot encoding.
        drop_background: Whether to drop the background class from the mask.

    Returns:
        A tuple containing:
        - The combined PyTorch tensor (C_total, H, W).
        - The total number of channels in the combined tensor.
    """
    if image_slice.shape != (target_size, target_size):
        image_slice = resize(image_slice, (target_size, target_size), order=1, preserve_range=True, anti_aliasing=True)
    if mask_slice.shape != (target_size, target_size):
        mask_slice = resize(mask_slice, (target_size, target_size), order=0, preserve_range=True, anti_aliasing=False)
        mask_slice = mask_slice.astype(np.int64)
    # Validate shape
    if image_slice.ndim!=2 or mask_slice.ndim!=2:
        raise ValueError(f"Expected 2D arrays for slices. "
                         f"Got image shape {image_slice.shape} and mask shape {mask_slice.shape}")
    if image_slice.shape!=mask_slice.shape:
        raise ValueError(f"Mismatched shape for image_slice,{image_slice.shape} and mask_slice, {mask_slice.shape}")
    
    # Process image and mask
    normalised_image = normalise_to_model_range(image_slice)
    image_tensor = torch.from_numpy(normalised_image).unsqueeze(0).float() # Shape: (1, H, W)

    one_hot_mask = one_hot_encode_mask(mask_slice, num_mask_classes, drop_background)
    combined_tensor = torch.cat([image_tensor, one_hot_mask], dim=0)
    total_channels = combined_tensor.shape[0]

    return combined_tensor, total_channels







