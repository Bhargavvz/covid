"""
3D U-Net model for lung segmentation using MONAI.

Configurable encoder/decoder architecture with skip connections
for binary lung mask prediction from CT volumes.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional


class UNet3D(nn.Module):
    """
    3D U-Net for volumetric lung segmentation.
    
    Uses MONAI's UNet implementation when available, with a pure PyTorch
    fallback for environments without MONAI.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        channels: Tuple[int, ...] = (32, 64, 128, 256, 512),
        strides: Tuple[int, ...] = (2, 2, 2, 2),
        num_res_units: int = 2,
        dropout: float = 0.1,
        use_monai: bool = True,
    ):
        """
        Args:
            in_channels: Number of input channels (1 for CT).
            out_channels: Number of output channels (1 for binary mask).
            channels: Feature channels at each encoder level.
            strides: Downsampling strides between levels.
            num_res_units: Number of residual units per level.
            dropout: Dropout probability.
            use_monai: Whether to use MONAI's UNet implementation.
        """
        super().__init__()
        self.use_monai = use_monai

        if use_monai:
            try:
                from monai.networks.nets import UNet
                self.model = UNet(
                    spatial_dims=3,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    channels=channels,
                    strides=strides,
                    num_res_units=num_res_units,
                    dropout=dropout,
                )
            except ImportError:
                self.use_monai = False

        if not self.use_monai:
            # Pure PyTorch fallback
            self.model = _PyTorchUNet3D(
                in_channels=in_channels,
                out_channels=out_channels,
                channels=channels,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor (B, C, D, H, W).

        Returns:
            Logits tensor (B, out_channels, D, H, W).
        """
        return self.model(x)

    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """
        Predict binary mask with sigmoid + threshold.

        Args:
            x: Input tensor (B, 1, D, H, W).
            threshold: Binarization threshold.

        Returns:
            Binary mask tensor (B, 1, D, H, W).
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.sigmoid(logits)
            masks = (probs > threshold).float()
        return masks


class _ConvBlock3D(nn.Module):
    """Double convolution block: Conv3D → BN → ReLU × 2."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class _PyTorchUNet3D(nn.Module):
    """Pure PyTorch 3D U-Net fallback (no MONAI dependency)."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        channels: Tuple[int, ...] = (32, 64, 128, 256, 512),
    ):
        super().__init__()

        # Encoder
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        prev_ch = in_channels
        for ch in channels[:-1]:
            self.encoders.append(_ConvBlock3D(prev_ch, ch))
            self.pools.append(nn.MaxPool3d(2))
            prev_ch = ch

        # Bottleneck
        self.bottleneck = _ConvBlock3D(channels[-2], channels[-1])

        # Decoder
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for i in range(len(channels) - 2, -1, -1):
            self.upconvs.append(
                nn.ConvTranspose3d(channels[i + 1], channels[i], 2, stride=2)
            )
            self.decoders.append(_ConvBlock3D(channels[i] * 2, channels[i]))

        # Final output
        self.final_conv = nn.Conv3d(channels[0], out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder path
        skip_connections = []
        for encoder, pool in zip(self.encoders, self.pools):
            x = encoder(x)
            skip_connections.append(x)
            x = pool(x)

        x = self.bottleneck(x)

        # Decoder path
        skip_connections.reverse()
        for upconv, decoder, skip in zip(self.upconvs, self.decoders, skip_connections):
            x = upconv(x)
            # Handle size mismatches
            if x.shape != skip.shape:
                x = nn.functional.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
            x = decoder(x)

        return self.final_conv(x)


if __name__ == "__main__":
    # Verify model with synthetic input
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet3D(in_channels=1, out_channels=1, use_monai=False).to(device)

    x = torch.randn(1, 1, 64, 64, 64, device=device)
    out = model(x)
    print(f"Input: {x.shape} → Output: {out.shape}")

    mask = model.predict(x)
    print(f"Predicted mask shape: {mask.shape}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    print("UNet3D module loaded successfully.")
