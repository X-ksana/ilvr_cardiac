import math
import random
from typing import Tuple, Optional, List, TYPE_CHECKING


from PIL import Image
import blobfile as bf
from mpi4py import MPI
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch
import os
import nibabel as nib # for loading nifti files
from .data_preprocessing_helpers import load_and_process_npy_pair,normalise_to_model_range
from torchvision import transforms
from . import dist_util

if TYPE_CHECKING:
    from typing import Tuple
    
    # This prevents circular import issues
    class ImageDataset(Dataset):
        pass







def load_data(
    *,
    data_dir:str,
    batch_size:int,
    image_size:int, # for resizing not cropping
    class_cond:bool=False,
    deterministic:bool=False,
    random_crop:bool=False,
    random_flip:bool=False,
    mask_dir:Optional[str]=None, # add mask dir
    num_mask_classes:int=4
) -> Tuple[DataLoader, 'ImageDataset']:
    """
    We need support for image-mask pairs as input
    For sampling, only refernce image is needed
    For training, training image-mask pairs are needed
    Image-mask pairs are stored in a single directory
    Our image is single channel, mask is single channel
    Need to concatenate them

    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.
    
    :param mask_dir: optional directory containing mask files with matching names.
    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    :param random_crop: if True, randomly crop the images for augmentation.
    :param random_flip: if True, randomly flip the images for augmentation.
    """
    allowed_extensions = ['.npy']

    if not data_dir:
        raise ValueError("unspecified data directory")
    all_files = _list_image_files_recursively(data_dir, allowed_extensions)
    
   # allowed_extensions = ['.npy']
    all_image_files = _list_image_files_recursively(data_dir, allowed_extensions)
    all_image_files = [f for f in all_image_files if '_seg' not in bf.basename(f)]

   # all_files = _list_image_files_recursively(data_dir, allowed_extensions)
    
    # If mask_dir provided, verifying matching mask files exists
    
    mask_map = None
    # Create a mapping of key and value for images and masks
    if mask_dir:
        print(f"Mask directory provided. Creating image-to-mask mapping...")
        mask_map = {}
        for img_path in all_image_files:
           # img_name_base = os.path.basename(img_path).split('.')[0]
            img_name_base = bf.basename(img_path).split('.')[0]
            mask_path = bf.join(mask_dir, f"{img_name_base}_seg.npy")
            if bf.exists(mask_path):
                mask_map[img_path] = mask_path
            else:
                print(f"Warning: Missing mask file for {img_name_base}. Skipping this image.")

        all_image_files = sorted(list(mask_map.keys()))
        print(f"Mapping complete. Found {len(all_image_files)} image-mask pairs.")


    # Not using classes for now, kiv with pathology labels
    classes = None
    if class_cond:
        # Assume classes are the first part of the filename,
        # before an underscore.
        class_names = [bf.basename(path).split("_")[0] for path in all_files]
        sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
        classes = [sorted_classes[x] for x in class_names]
    
    dataset = ImageDataset(
        resolution=image_size,
        image_paths=all_image_files, # Use only image files
        mask_map=mask_map, # Add mask_map
        num_mask_classes=num_mask_classes,
        classes=classes,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size(),
        random_crop=random_crop,
        random_flip=random_flip,
    )
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
        )
    while True:
        yield from loader


def _list_image_files_recursively(data_dir:str,allowed_extensions: List[str]) -> List[str]:
    results = []
    for entry in sorted(os.listdir(data_dir)):
        full_path = os.path.join(data_dir, entry)
        if os.path.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path, allowed_extensions))
        elif any(full_path.lower().endswith(ext) for ext in allowed_extensions):
            results.append(full_path)
    return results


  

class ImageDataset(Dataset):
    def __init__(
        self,
        resolution:int,
        image_paths:List[str],
        mask_map:Optional[dict] = None, # Add mask_map
        num_mask_classes:int = 4,
        classes:Optional[List]=None,
        shard:int = 0,
        num_shards:int =1,
        random_crop:bool = False,
        random_flip: bool = False,
    ):
        super().__init__()
        self.resolution = resolution
        self.mask_map = mask_map
      #  self.local_images = image_paths[shard:][::num_shards]
        self.local_classes = None if classes is None else classes[shard:][::num_shards]
        self.random_crop = random_crop
        self.random_flip = random_flip
        self.num_mask_classes = num_mask_classes

    # Build a list of all slices from all files
        self.samples = []
        print("Scanning npy to build the slice index")
        for img_path in image_paths:
            try:
                # Load the first channel 
                shape = np.load(img_path, mmap_mode='r').shape
                # Expected shape (1,Num Slices,H,W)
                if len(shape)!=4 or shape[0]!=1:
                    print(f"Warning: Skipping file with unexpected shape: {img_path} - shape {shape}")
                    continue
                number_slices = shape[1]
                for i in range(number_slices):
                    self.samples.append((img_path,i))
            except Exception as e:
                print(f"Warning: Could not read shape from image path {img_path}. Skipping. Error: {e}")

        print(f"Found {len(self.samples)} total slices from across {len(image_paths)} files.")

        # Apply shard
        self.local_samples = self.samples[shard:][::num_shards]


    def __len__(self):
        return len(self.local_samples)

    def __getitem__(self, idx:int) -> Tuple[torch.Tensor,dict]:
        image_path, slice_idx = self.local_samples[idx]
        out_dict = {}

        if self.mask_map:
            # Paied image-mask
            mask_path = self.mask_map[image_path]

            # Load the full 3D volumes
            full_image_volume = np.load(image_path) # Shape: (1, D, H, W)
            full_mask_volume = np.load(mask_path) # Shape: (1 ,D ,H , W)
           
            image_slice = full_image_volume[0,slice_idx,:,:]
            mask_slice = full_mask_volume[0,slice_idx,:,:]

            # Use helper function to process
            arr, _ = load_and_process_npy_pair(
                image_slice, mask_slice, self.num_mask_classes,target_size=self.resolution
            )
        else:
            # Path for Image-Only Data (if needed) ---
            full_image_volume = np.load(image_path)
            image_slice = full_image_volume[0, slice_idx, :, :]
            normalised_slice = normalise_to_model_range(image_slice)
            arr = torch.from_numpy(normalised_slice).unsqueeze(0).float()

        # Random flipping can be applied to the final 2D tensor slice
        if self.random_flip and random.random() < 0.5:
            arr = torch.flip(arr, dims=[-1]) # Flips horizontally

        if self.local_classes is not None:
            # This would need adjustment if classes are per-slice
            out_dict["y"] = "class_logic_goes_here"

        return arr, out_dict


def center_crop_arr(pil_image, image_size):
    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]


def random_crop_arr(pil_image, image_size, min_crop_frac=0.8, max_crop_frac=1.0):
    min_smaller_dim_size = math.ceil(image_size / max_crop_frac)
    max_smaller_dim_size = math.ceil(image_size / min_crop_frac)
    smaller_dim_size = random.randrange(min_smaller_dim_size, max_smaller_dim_size + 1)

    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * smaller_dim_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = smaller_dim_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = random.randrange(arr.shape[0] - image_size + 1)
    crop_x = random.randrange(arr.shape[1] - image_size + 1)
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]


# Need to add load single-channel reference image, add a noise channel for the mask (for png)
def load_reference_data_png(data_dir, batch_size, image_size):
    """
    Loads a reference image, normalizes it, and adds a noise channel for the mask.
    This prepares a 2-channel tensor that matches the model's training input.
    """
    if not data_dir:
        raise ValueError("A reference directory must be provided with --ref_dir")
    
    all_files = _list_image_files_recursively(data_dir)
    
    # Define a transform to resize and normalize the reference images
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)), # Normalizes single channel to [-1, 1]
    ])

    # Create batches of reference images
    for i in range(0, len(all_files), batch_size):
        batch_files = all_files[i:i+batch_size]
        ref_images = []
        for path in batch_files:
            with open(path, "rb") as f:
                # Open as grayscale ('L')
                img = Image.open(f).convert("L")
                ref_images.append(transform(img))

        # Stack single-channel images into a batch tensor
        ref_image_tensor = torch.stack(ref_images, dim=0)
        
        # Create a noise channel of the same shape for the mask
        ref_mask_noise = torch.randn_like(ref_image_tensor)
        
        # Concatenate to create the 2-channel reference tensor (image + noisy mask)
        ref_tensor = torch.cat([ref_image_tensor, ref_mask_noise], dim=1)
        
        # Prepare the model_kwargs dictionary
        model_kwargs = {}
        model_kwargs["ref_img"] = ref_tensor.to(dist_util.dev())
        yield model_kwargs