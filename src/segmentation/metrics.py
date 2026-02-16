"""
Segmentation metrics: Dice score, IoU, and Hausdorff distance.
"""

import torch
import numpy as np
from typing import Optional


def dice_score(
    pred: torch.Tensor,
    target: torch.Tensor,
    smooth: float = 1e-5,
    threshold: Optional[float] = 0.5,
) -> torch.Tensor:
    """
    Compute Dice similarity coefficient.

    Args:
        pred: Predicted logits or probabilities (B, C, D, H, W).
        target: Ground truth binary mask (B, C, D, H, W).
        smooth: Smoothing factor.
        threshold: If not None, binarize predictions.

    Returns:
        Mean Dice score (scalar tensor).
    """
    if threshold is not None:
        pred = (torch.sigmoid(pred) > threshold).float()

    intersection = (pred * target).sum(dim=(2, 3, 4))
    union = pred.sum(dim=(2, 3, 4)) + target.sum(dim=(2, 3, 4))

    dice = (2.0 * intersection + smooth) / (union + smooth)
    return dice.mean()


def iou_score(
    pred: torch.Tensor,
    target: torch.Tensor,
    smooth: float = 1e-5,
    threshold: Optional[float] = 0.5,
) -> torch.Tensor:
    """
    Compute Intersection over Union (Jaccard index).

    Args:
        pred: Predicted tensor (B, C, D, H, W).
        target: Ground truth tensor (B, C, D, H, W).
        smooth: Smoothing factor.
        threshold: Binarization threshold.

    Returns:
        Mean IoU score (scalar tensor).
    """
    if threshold is not None:
        pred = (torch.sigmoid(pred) > threshold).float()

    intersection = (pred * target).sum(dim=(2, 3, 4))
    union = pred.sum(dim=(2, 3, 4)) + target.sum(dim=(2, 3, 4)) - intersection

    iou = (intersection + smooth) / (union + smooth)
    return iou.mean()


def hausdorff_distance_95(
    pred: np.ndarray,
    target: np.ndarray,
    voxel_spacing: tuple = (1.0, 1.0, 1.0),
) -> float:
    """
    Compute 95th percentile Hausdorff distance between binary masks.

    Args:
        pred: Predicted binary mask (D, H, W).
        target: Ground truth binary mask (D, H, W).
        voxel_spacing: Physical voxel dimensions.

    Returns:
        95th percentile Hausdorff distance in mm.
    """
    from scipy.ndimage import distance_transform_edt

    pred = pred.astype(bool)
    target = target.astype(bool)

    if not pred.any() or not target.any():
        return float("inf")

    # Distance transform of the complement
    dt_pred = distance_transform_edt(~pred, sampling=voxel_spacing)
    dt_target = distance_transform_edt(~target, sampling=voxel_spacing)

    # Surface distances
    surface_pred = dt_target[pred]
    surface_target = dt_pred[target]

    all_distances = np.concatenate([surface_pred, surface_target])
    return float(np.percentile(all_distances, 95))


class DiceBCELoss(torch.nn.Module):
    """Combined Dice + Binary Cross-Entropy loss for segmentation."""

    def __init__(self, dice_weight: float = 0.5, bce_weight: float = 0.5, smooth: float = 1e-5):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.smooth = smooth
        self.bce = torch.nn.BCEWithLogitsLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # BCE component
        bce_loss = self.bce(pred, target)

        # Dice component
        pred_sigmoid = torch.sigmoid(pred)
        intersection = (pred_sigmoid * target).sum(dim=(2, 3, 4))
        union = pred_sigmoid.sum(dim=(2, 3, 4)) + target.sum(dim=(2, 3, 4))
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1.0 - dice.mean()

        return self.dice_weight * dice_loss + self.bce_weight * bce_loss


if __name__ == "__main__":
    # Test metrics
    pred = torch.randn(2, 1, 32, 32, 32)
    target = (torch.randn(2, 1, 32, 32, 32) > 0).float()

    d = dice_score(pred, target)
    i = iou_score(pred, target)
    print(f"Dice: {d:.4f}, IoU: {i:.4f}")

    loss_fn = DiceBCELoss()
    loss = loss_fn(pred, target)
    print(f"DiceBCE Loss: {loss:.4f}")
    print("Metrics module loaded successfully.")
