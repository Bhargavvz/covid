"""
CT volume preprocessing transforms.

Includes resampling, window/level normalization, resizing,
and data augmentation using MONAI transforms.
"""

import logging
from typing import Tuple, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

# Default CT lung window parameters
DEFAULT_HU_MIN = -1000.0
DEFAULT_HU_MAX = 400.0
DEFAULT_VOLUME_SIZE = (128, 128, 128)


def resample_volume(
    volume: np.ndarray,
    original_spacing: Sequence[float],
    target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> np.ndarray:
    """
    Resample a 3D volume to isotropic target spacing.

    Args:
        volume: 3D numpy array (D, H, W).
        original_spacing: Original voxel spacing (z, y, x).
        target_spacing: Target voxel spacing.

    Returns:
        Resampled 3D volume.
    """
    from scipy.ndimage import zoom

    original_spacing = np.array(original_spacing, dtype=np.float64)
    target_spacing = np.array(target_spacing, dtype=np.float64)

    resize_factor = original_spacing / target_spacing
    new_shape = np.round(np.array(volume.shape) * resize_factor).astype(int)

    resampled = zoom(volume, resize_factor, order=3, mode="nearest")
    logger.debug(
        f"Resampled from {volume.shape} (spacing={original_spacing.tolist()}) "
        f"to {resampled.shape} (spacing={target_spacing})"
    )
    return resampled.astype(np.float32)


def apply_lung_window(
    volume: np.ndarray,
    hu_min: float = DEFAULT_HU_MIN,
    hu_max: float = DEFAULT_HU_MAX,
) -> np.ndarray:
    """
    Apply lung window clipping and normalize to [0, 1].

    Args:
        volume: 3D volume in Hounsfield Units.
        hu_min: Lower HU bound (default: -1000).
        hu_max: Upper HU bound (default: 400).

    Returns:
        Normalized volume in [0, 1].
    """
    volume = np.clip(volume, hu_min, hu_max)
    volume = (volume - hu_min) / (hu_max - hu_min)
    return volume.astype(np.float32)


def resize_volume(
    volume: np.ndarray,
    target_size: Tuple[int, int, int] = DEFAULT_VOLUME_SIZE,
) -> np.ndarray:
    """
    Resize a 3D volume to a fixed target size using trilinear interpolation.

    Args:
        volume: 3D numpy array.
        target_size: Target dimensions (D, H, W).

    Returns:
        Resized volume.
    """
    from scipy.ndimage import zoom

    current_shape = np.array(volume.shape, dtype=np.float64)
    target_shape = np.array(target_size, dtype=np.float64)
    zoom_factors = target_shape / current_shape

    resized = zoom(volume, zoom_factors, order=1, mode="nearest")
    return resized.astype(np.float32)


def preprocess_volume(
    volume: np.ndarray,
    spacing: Optional[Sequence[float]] = None,
    target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    hu_min: float = DEFAULT_HU_MIN,
    hu_max: float = DEFAULT_HU_MAX,
    target_size: Tuple[int, int, int] = DEFAULT_VOLUME_SIZE,
) -> np.ndarray:
    """
    Full preprocessing pipeline: resample → window → resize.

    Args:
        volume: Raw 3D CT volume (HU values).
        spacing: Original voxel spacing. If None, skip resampling.
        target_spacing: Target isotropic spacing.
        hu_min: Lower HU bound for windowing.
        hu_max: Upper HU bound for windowing.
        target_size: Final volume dimensions.

    Returns:
        Preprocessed volume (D, H, W) in [0, 1].
    """
    if spacing is not None:
        volume = resample_volume(volume, spacing, target_spacing)

    volume = apply_lung_window(volume, hu_min, hu_max)
    volume = resize_volume(volume, target_size)

    return volume


def get_train_transforms(
    volume_size: Tuple[int, int, int] = DEFAULT_VOLUME_SIZE,
    keys: Tuple[str, ...] = ("image",),
):
    """
    Get MONAI training augmentation transforms.

    Includes:
    - Random affine (rotation, scaling)
    - Random elastic deformation
    - Random Gaussian noise
    - Random intensity shift/scale  
    - Ensure channel dimension

    Args:
        volume_size: Expected input volume size.
        keys: Dictionary keys for MONAI transforms.

    Returns:
        MONAI Compose transform.
    """
    from monai.transforms import (
        Compose,
        EnsureChannelFirstd,
        RandAffined,
        RandGaussianNoised,
        RandAdjustContrastd,
        RandScaleIntensityd,
        RandShiftIntensityd,
        ToTensord,
    )

    return Compose([
        EnsureChannelFirstd(keys=keys),
        RandAffined(
            keys=keys,
            prob=0.5,
            rotate_range=(0.26, 0.26, 0.26),  # ~15 degrees
            scale_range=(0.1, 0.1, 0.1),
            mode="bilinear" if "label" not in keys else ("bilinear", "nearest"),
            padding_mode="zeros",
        ),
        RandGaussianNoised(keys=("image",) if "image" in keys else keys[:1], prob=0.3, mean=0.0, std=0.02),
        RandScaleIntensityd(keys=("image",) if "image" in keys else keys[:1], prob=0.3, factors=0.1),
        RandShiftIntensityd(keys=("image",) if "image" in keys else keys[:1], prob=0.3, offsets=0.05),
        ToTensord(keys=keys),
    ])


def get_val_transforms(
    volume_size: Tuple[int, int, int] = DEFAULT_VOLUME_SIZE,
    keys: Tuple[str, ...] = ("image",),
):
    """
    Get MONAI validation transforms (no augmentation).

    Args:
        volume_size: Expected input volume size.
        keys: Dictionary keys.

    Returns:
        MONAI Compose transform.
    """
    from monai.transforms import (
        Compose,
        EnsureChannelFirstd,
        ToTensord,
    )

    return Compose([
        EnsureChannelFirstd(keys=keys),
        ToTensord(keys=keys),
    ])


if __name__ == "__main__":
    # Quick verification with synthetic data
    vol = np.random.uniform(-1000, 400, size=(64, 256, 256)).astype(np.float32)
    processed = preprocess_volume(vol, spacing=None)
    print(f"Input shape: (64, 256, 256) → Output shape: {processed.shape}")
    print(f"Value range: [{processed.min():.3f}, {processed.max():.3f}]")
    print("Transforms module loaded successfully.")
