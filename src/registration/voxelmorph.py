"""
VoxelMorph 3D — Deep learning-based deformable image registration.

Implements a U-Net encoder-decoder that predicts a dense 3D deformation field
for aligning a moving image to a fixed image, with a spatial transformer
for applying the deformation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class SpatialTransformer3D(nn.Module):
    """
    3D Spatial Transformer Network module.
    
    Applies a 3D deformation field to warp an input volume using
    bilinear interpolation.
    """

    def __init__(self, size: Tuple[int, int, int]):
        """
        Args:
            size: Spatial dimensions (D, H, W) of the expected input.
        """
        super().__init__()

        # Create normalized identity grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors, indexing="ij")
        grid = torch.stack(grids)  # (3, D, H, W)
        grid = grid.float().unsqueeze(0)  # (1, 3, D, H, W)

        # Normalize to [-1, 1] for grid_sample
        for i in range(3):
            grid[:, i] = 2.0 * grid[:, i] / (size[i] - 1) - 1.0

        self.register_buffer("grid", grid)

    def forward(
        self, src: torch.Tensor, flow: torch.Tensor
    ) -> torch.Tensor:
        """
        Warp source image using deformation field.

        Args:
            src: Source/moving image (B, C, D, H, W).
            flow: Deformation field (B, 3, D, H, W).

        Returns:
            Warped image (B, C, D, H, W).
        """
        # Normalize flow to [-1, 1] range
        shape = src.shape[2:]
        flow_normalized = flow.clone()
        for i in range(3):
            flow_normalized[:, i] = 2.0 * flow[:, i] / (shape[i] - 1)

        # Add flow to identity grid
        new_grid = self.grid + flow_normalized

        # Rearrange for grid_sample: (B, D, H, W, 3)
        new_grid = new_grid.permute(0, 2, 3, 4, 1)

        return F.grid_sample(
            src, new_grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )


class ConvBlock(nn.Module):
    """Convolution block: Conv3D → LeakyReLU."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, stride=stride, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class VoxelMorph3D(nn.Module):
    """
    VoxelMorph 3D registration network.

    Architecture:
    - U-Net encoder-decoder takes concatenated (moving, fixed) images as input
    - Outputs a 3D deformation field
    - Spatial transformer warps the moving image
    """

    def __init__(
        self,
        volume_size: Tuple[int, int, int] = (128, 128, 128),
        enc_channels: Tuple[int, ...] = (16, 32, 32, 32, 32),
        dec_channels: Tuple[int, ...] = (32, 32, 32, 32, 16, 16),
        in_channels: int = 2,  # moving + fixed concatenated
    ):
        """
        Args:
            volume_size: Expected input spatial dimensions (D, H, W).
            enc_channels: Encoder feature channels per level.
            dec_channels: Decoder feature channels per level.
            in_channels: Number of input channels (2 for concatenated pair).
        """
        super().__init__()
        self.volume_size = volume_size

        # Encoder
        self.encoders = nn.ModuleList()
        prev_ch = in_channels
        for ch in enc_channels:
            self.encoders.append(ConvBlock(prev_ch, ch, stride=2))
            prev_ch = ch

        # Decoder
        self.decoders = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        enc_idx = len(enc_channels) - 1
        for i, ch in enumerate(dec_channels):
            # Upsample
            self.upsamples.append(
                nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)
            )
            # Skip connection adds encoder channels
            if i < len(enc_channels):
                skip_ch = enc_channels[enc_idx] if enc_idx >= 0 else 0
                enc_idx -= 1
                in_ch = prev_ch + skip_ch
            else:
                in_ch = prev_ch
            self.decoders.append(ConvBlock(in_ch, ch))
            prev_ch = ch

        # Flow prediction (3D deformation field)
        self.flow = nn.Conv3d(dec_channels[-1], 3, 3, padding=1)
        # Initialize flow weights to small values
        self.flow.weight.data.normal_(0, 1e-5)
        self.flow.bias.data.zero_()

        # Spatial transformer
        self.spatial_transformer = SpatialTransformer3D(volume_size)

    def forward(
        self,
        moving: torch.Tensor,
        fixed: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            moving: Moving image (B, 1, D, H, W).
            fixed: Fixed image (B, 1, D, H, W).

        Returns:
            moved: Warped moving image (B, 1, D, H, W).
            flow: Deformation field (B, 3, D, H, W).
        """
        # Concatenate inputs
        x = torch.cat([moving, fixed], dim=1)  # (B, 2, D, H, W)

        # Encoder
        skip_connections = []
        for encoder in self.encoders:
            x = encoder(x)
            skip_connections.append(x)

        # Decoder with skip connections
        skip_connections.reverse()
        for i, (upsample, decoder) in enumerate(zip(self.upsamples, self.decoders)):
            x = upsample(x)
            if i < len(skip_connections):
                skip = skip_connections[i]
                # Handle size mismatches
                if x.shape[2:] != skip.shape[2:]:
                    x = F.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=True)
                x = torch.cat([x, skip], dim=1)
            x = decoder(x)

        # Predict flow
        flow = self.flow(x)

        # Resize flow to match input size if needed
        if flow.shape[2:] != moving.shape[2:]:
            flow = F.interpolate(flow, size=moving.shape[2:], mode="trilinear", align_corners=True)

        # Warp moving image
        moved = self.spatial_transformer(moving, flow)

        return moved, flow


class RegistrationLoss(nn.Module):
    """
    Combined registration loss: similarity + regularization.
    
    Similarity: NCC (Normalized Cross-Correlation) or MSE
    Regularization: Gradient smoothness of deformation field
    """

    def __init__(
        self,
        similarity: str = "ncc",
        reg_weight: float = 1.0,
        window_size: int = 9,
    ):
        """
        Args:
            similarity: 'ncc' or 'mse'.
            reg_weight: Weight for regularization loss.
            window_size: Window size for NCC computation.
        """
        super().__init__()
        self.similarity = similarity
        self.reg_weight = reg_weight
        self.window_size = window_size

    def ncc_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute local normalized cross-correlation loss (NaN-safe)."""
        ndim = 3
        window = [self.window_size] * ndim
        sum_filt = torch.ones([1, 1, *window], device=pred.device, dtype=pred.dtype)
        pad_size = [w // 2 for w in window]

        stride = [1] * ndim
        padding = pad_size

        # Compute local sums
        pred_sum = F.conv3d(pred, sum_filt, stride=stride, padding=padding)
        target_sum = F.conv3d(target, sum_filt, stride=stride, padding=padding)
        pred_sq_sum = F.conv3d(pred * pred, sum_filt, stride=stride, padding=padding)
        target_sq_sum = F.conv3d(target * target, sum_filt, stride=stride, padding=padding)
        pred_target_sum = F.conv3d(pred * target, sum_filt, stride=stride, padding=padding)

        win_size = float(torch.prod(torch.tensor(window)))
        pred_mean = pred_sum / win_size
        target_mean = target_sum / win_size

        cross = pred_target_sum - target_mean * pred_sum - pred_mean * target_sum + pred_mean * target_mean * win_size
        pred_var = pred_sq_sum - 2 * pred_mean * pred_sum + pred_mean * pred_mean * win_size
        target_var = target_sq_sum - 2 * target_mean * target_sum + target_mean * target_mean * win_size

        # Clamp variances to prevent negative values from FP16 precision
        pred_var = torch.clamp(pred_var, min=0.0)
        target_var = torch.clamp(target_var, min=0.0)

        # Use larger epsilon for FP16 stability
        cc = cross * cross / (pred_var * target_var + 1e-3)
        cc = torch.clamp(cc, 0.0, 1.0)  # CC should be in [0, 1]

        return 1.0 - cc.mean()

    def gradient_loss(self, flow: torch.Tensor) -> torch.Tensor:
        """Compute gradient smoothness regularization."""
        dy = torch.abs(flow[:, :, 1:, :, :] - flow[:, :, :-1, :, :])
        dx = torch.abs(flow[:, :, :, 1:, :] - flow[:, :, :, :-1, :])
        dz = torch.abs(flow[:, :, :, :, 1:] - flow[:, :, :, :, :-1])
        return (dy.mean() + dx.mean() + dz.mean()) / 3.0

    def forward(
        self,
        moved: torch.Tensor,
        fixed: torch.Tensor,
        flow: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute total loss.

        Returns:
            loss: Total loss scalar.
            components: Dict with individual loss components.
        """
        if self.similarity == "ncc":
            sim_loss = self.ncc_loss(moved, fixed)
        else:
            sim_loss = F.mse_loss(moved, fixed)

        reg_loss = self.gradient_loss(flow)

        total_loss = sim_loss + self.reg_weight * reg_loss

        return total_loss, {
            "similarity": sim_loss.item(),
            "regularization": reg_loss.item(),
            "total": total_loss.item(),
        }


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vol_size = (64, 64, 64)

    model = VoxelMorph3D(volume_size=vol_size).to(device)
    moving = torch.randn(1, 1, *vol_size, device=device)
    fixed = torch.randn(1, 1, *vol_size, device=device)

    moved, flow = model(moving, fixed)
    print(f"Moving: {moving.shape}")
    print(f"Fixed:  {fixed.shape}")
    print(f"Moved:  {moved.shape}")
    print(f"Flow:   {flow.shape}")

    criterion = RegistrationLoss(similarity="ncc")
    loss, components = criterion(moved, fixed, flow)
    print(f"Loss: {loss.item():.4f} (sim: {components['similarity']:.4f}, reg: {components['regularization']:.4f})")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}")
    print("VoxelMorph3D module loaded successfully.")
