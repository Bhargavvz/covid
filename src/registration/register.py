"""
Image registration inference utilities.

Provides functions for:
- Registering baseline and follow-up scans using VoxelMorph or SimpleITK
- Computing difference maps between registered scan pairs
- Visualizing deformation fields
"""

import logging
from typing import Tuple, Optional, Dict, Any

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def register_scans(
    moving: np.ndarray,
    fixed: np.ndarray,
    model: Optional[torch.nn.Module] = None,
    device: Optional[torch.device] = None,
    method: str = "voxelmorph",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Register a moving scan to a fixed scan.

    Args:
        moving: Moving volume (D, H, W), normalized [0,1].
        fixed: Fixed volume (D, H, W), normalized [0,1].
        model: VoxelMorph model (required for 'voxelmorph' method).
        device: Torch device.
        method: 'voxelmorph' or 'simpleitk'.

    Returns:
        registered: The registered (warped) moving image.
        deformation: The deformation field (3, D, H, W) or None.
    """
    if method == "voxelmorph":
        return _register_voxelmorph(moving, fixed, model, device)
    elif method == "simpleitk":
        return _register_simpleitk(moving, fixed)
    else:
        raise ValueError(f"Unknown registration method: {method}")


def _register_voxelmorph(
    moving: np.ndarray,
    fixed: np.ndarray,
    model: torch.nn.Module,
    device: Optional[torch.device] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Register using VoxelMorph deep learning model."""
    if model is None:
        raise ValueError("VoxelMorph model is required for deep learning registration")

    if device is None:
        device = next(model.parameters()).device

    # Convert to tensors: (1, 1, D, H, W)
    moving_t = torch.from_numpy(moving).float().unsqueeze(0).unsqueeze(0).to(device)
    fixed_t = torch.from_numpy(fixed).float().unsqueeze(0).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        moved_t, flow_t = model(moving_t, fixed_t)

    registered = moved_t.squeeze().cpu().numpy()
    deformation = flow_t.squeeze().cpu().numpy()  # (3, D, H, W)

    logger.info(f"VoxelMorph registration complete. Flow range: [{deformation.min():.3f}, {deformation.max():.3f}]")

    return registered, deformation


def _register_simpleitk(
    moving: np.ndarray,
    fixed: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Register using SimpleITK B-spline deformable registration."""
    try:
        import SimpleITK as sitk
    except ImportError:
        raise ImportError("SimpleITK is required: pip install SimpleITK")

    # Convert numpy to SimpleITK images
    fixed_img = sitk.GetImageFromArray(fixed.astype(np.float64))
    moving_img = sitk.GetImageFromArray(moving.astype(np.float64))

    # BSpline registration
    registration = sitk.ImageRegistrationMethod()

    # Similarity metric
    registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration.SetMetricSamplingStrategy(registration.RANDOM)
    registration.SetMetricSamplingPercentage(0.1)

    # Optimizer
    registration.SetOptimizerAsGradientDescent(
        learningRate=1.0,
        numberOfIterations=200,
        convergenceMinimumValue=1e-6,
        convergenceWindowSize=10,
    )

    # BSpline transform
    grid_physical_spacing = [50.0, 50.0, 50.0]
    image_physical_size = [
        fixed_img.GetSize()[i] * fixed_img.GetSpacing()[i]
        for i in range(3)
    ]
    mesh_size = [
        int(image_physical_size[i] / grid_physical_spacing[i] + 0.5)
        for i in range(3)
    ]

    initial_transform = sitk.BSplineTransformInitializer(
        fixed_img, mesh_size, order=3
    )
    registration.SetInitialTransform(initial_transform)

    # Multi-resolution
    registration.SetShrinkFactorsPerLevel([4, 2, 1])
    registration.SetSmoothingSigmasPerLevel([4, 2, 0])
    registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    # Interpolator
    registration.SetInterpolator(sitk.sitkLinear)

    logger.info("Running SimpleITK BSpline registration...")
    final_transform = registration.Execute(fixed_img, moving_img)

    # Apply transform
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed_img)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0)
    resampler.SetTransform(final_transform)
    registered_img = resampler.Execute(moving_img)

    registered = sitk.GetArrayFromImage(registered_img).astype(np.float32)

    # Compute displacement field
    displacement_filter = sitk.TransformToDisplacementFieldFilter()
    displacement_filter.SetReferenceImage(fixed_img)
    disp_field = displacement_filter.Execute(final_transform)
    deformation = sitk.GetArrayFromImage(disp_field).astype(np.float32)
    deformation = np.transpose(deformation, (3, 0, 1, 2))  # (3, D, H, W)

    logger.info(f"SimpleITK registration complete. Iterations: {registration.GetOptimizerIteration()}")

    return registered, deformation


def compute_difference_map(
    baseline: np.ndarray,
    followup_registered: np.ndarray,
    method: str = "absolute",
) -> np.ndarray:
    """
    Compute difference map between baseline and registered follow-up.

    Args:
        baseline: Baseline volume (D, H, W).
        followup_registered: Registered follow-up volume (D, H, W).
        method: 'absolute' or 'signed'.

    Returns:
        Difference map (D, H, W).
    """
    diff = followup_registered - baseline

    if method == "absolute":
        diff = np.abs(diff)
    elif method == "signed":
        pass  # Keep as-is
    else:
        raise ValueError(f"Unknown difference method: {method}")

    # Normalize to [0, 1]
    diff_max = diff.max()
    if diff_max > 0:
        diff = diff / diff_max

    return diff.astype(np.float32)


def compute_deformation_magnitude(deformation: np.ndarray) -> np.ndarray:
    """
    Compute magnitude of deformation field.

    Args:
        deformation: Deformation field (3, D, H, W).

    Returns:
        Magnitude map (D, H, W).
    """
    magnitude = np.sqrt(np.sum(deformation ** 2, axis=0))
    return magnitude.astype(np.float32)


if __name__ == "__main__":
    # Test with synthetic data
    vol_size = (64, 64, 64)
    baseline = np.random.rand(*vol_size).astype(np.float32) * 0.5 + 0.25
    followup = baseline + np.random.rand(*vol_size).astype(np.float32) * 0.1

    diff_map = compute_difference_map(baseline, followup)
    print(f"Difference map shape: {diff_map.shape}, range: [{diff_map.min():.3f}, {diff_map.max():.3f}]")

    deformation = np.random.randn(3, *vol_size).astype(np.float32) * 2
    magnitude = compute_deformation_magnitude(deformation)
    print(f"Deformation magnitude shape: {magnitude.shape}")
    print("Registration module loaded successfully.")
