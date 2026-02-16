"""
3D ResNet for Post-COVID severity classification and lung damage regression.

Multi-task architecture:
- Head 1: 4-class severity classification (Normal/Mild/Moderate/Severe)
- Head 2: Regression (% lung involvement)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict


class ResidualBlock3D(nn.Module):
    """3D Residual block with optional downsampling."""

    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, downsample: Optional[nn.Module] = None):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)


class Bottleneck3D(nn.Module):
    """3D Bottleneck block for deeper networks."""

    expansion = 4

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, downsample: Optional[nn.Module] = None):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.conv3 = nn.Conv3d(out_channels, out_channels * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm3d(out_channels * self.expansion)
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)


class ResNet3D(nn.Module):
    """
    3D ResNet for CT volume classification with multi-task heads.

    Supports ResNet-18, ResNet-34, ResNet-50 configurations.
    
    Outputs:
        - severity_logits: (B, num_classes) for classification
        - damage_pct: (B, 1) for regression (% lung involvement)
    """

    CONFIGS = {
        18: (ResidualBlock3D, [2, 2, 2, 2]),
        34: (ResidualBlock3D, [3, 4, 6, 3]),
        50: (Bottleneck3D, [3, 4, 6, 3]),
    }

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 4,
        depth: int = 50,
        dropout: float = 0.3,
        feature_dim: int = 512,
    ):
        """
        Args:
            in_channels: Input channels (1 for CT).
            num_classes: Number of severity classes.
            depth: ResNet depth (18, 34, or 50).
            dropout: Dropout probability.
            feature_dim: Feature dimension before heads.
        """
        super().__init__()

        if depth not in self.CONFIGS:
            raise ValueError(f"Unsupported depth {depth}. Choose from {list(self.CONFIGS.keys())}")

        block, layers = self.CONFIGS[depth]
        self.in_channels_current = 64

        # Stem
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(3, stride=2, padding=1),
        )

        # Residual layers
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.dropout = nn.Dropout(dropout)

        final_channels = 512 * block.expansion

        # Feature projection
        self.feature_proj = nn.Sequential(
            nn.Linear(final_channels, feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # Classification head (severity)
        self.classifier = nn.Linear(feature_dim, num_classes)

        # Regression head (% lung damage)
        self.regressor = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
            nn.Sigmoid(),  # Output in [0, 1], scale to [0, 100]
        )

        # Weight initialization
        self._initialize_weights()

    def _make_layer(self, block, out_channels: int, num_blocks: int, stride: int = 1) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.in_channels_current != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.in_channels_current, out_channels * block.expansion, 1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels * block.expansion),
            )

        layers = [block(self.in_channels_current, out_channels, stride, downsample)]
        self.in_channels_current = out_channels * block.expansion
        for _ in range(1, num_blocks):
            layers.append(block(self.in_channels_current, out_channels))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract feature vector from input volume."""
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.global_pool(x).flatten(1)
        x = self.feature_proj(x)
        return x

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass with multi-task output.

        Args:
            x: Input volume (B, 1, D, H, W).

        Returns:
            Dict with:
                - 'severity_logits': (B, num_classes)
                - 'damage_pct': (B, 1) in [0, 100]
                - 'features': (B, feature_dim)
        """
        features = self.extract_features(x)
        features_dropped = self.dropout(features)

        severity_logits = self.classifier(features_dropped)
        damage_pct = self.regressor(features_dropped) * 100  # Scale to percentage

        return {
            "severity_logits": severity_logits,
            "damage_pct": damage_pct,
            "features": features,
        }

    def predict(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Inference prediction with class probabilities.

        Returns:
            Dict with severity class, probability, damage percentage.
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(x)
            probs = F.softmax(outputs["severity_logits"], dim=1)
            severity_class = probs.argmax(dim=1)
            severity_prob = probs.max(dim=1).values

        return {
            "severity_class": severity_class,
            "severity_prob": severity_prob,
            "severity_probs": probs,
            "damage_pct": outputs["damage_pct"].squeeze(-1),
        }


class MultiTaskLoss(nn.Module):
    """
    Combined classification + regression loss for multi-task training.
    
    Uses:
    - Focal Loss for classification (handles class imbalance)
    - MSE for regression
    """

    def __init__(
        self,
        num_classes: int = 4,
        cls_weight: float = 1.0,
        reg_weight: float = 0.5,
        focal_gamma: float = 2.0,
        focal_alpha: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.cls_weight = cls_weight
        self.reg_weight = reg_weight
        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha
        self.mse = nn.MSELoss()

    def focal_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Focal loss for classification."""
        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal = ((1 - pt) ** self.focal_gamma) * ce_loss

        if self.focal_alpha is not None:
            alpha = self.focal_alpha.to(logits.device)
            alpha_t = alpha.gather(0, targets)
            focal = alpha_t * focal

        return focal.mean()

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: torch.Tensor,
        damage_targets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute multi-task loss.

        Args:
            outputs: Model outputs dict.
            targets: Classification targets (B,).
            damage_targets: Optional regression targets (B,) in [0, 100].

        Returns:
            total_loss: Combined loss.
            components: Dict of individual loss values.
        """
        cls_loss = self.focal_loss(outputs["severity_logits"], targets)

        reg_loss = torch.tensor(0.0, device=cls_loss.device)
        if damage_targets is not None:
            reg_loss = self.mse(outputs["damage_pct"].squeeze(-1), damage_targets)

        total = self.cls_weight * cls_loss + self.reg_weight * reg_loss

        return total, {
            "classification": cls_loss.item(),
            "regression": reg_loss.item(),
            "total": total.item(),
        }


SEVERITY_LABELS = {0: "Normal", 1: "Mild", 2: "Moderate", 3: "Severe"}


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for depth in [18, 34, 50]:
        model = ResNet3D(in_channels=1, num_classes=4, depth=depth).to(device)
        x = torch.randn(2, 1, 64, 64, 64, device=device)
        outputs = model(x)
        print(f"ResNet3D-{depth}:")
        print(f"  Severity logits: {outputs['severity_logits'].shape}")
        print(f"  Damage %: {outputs['damage_pct'].shape}")
        print(f"  Features: {outputs['features'].shape}")
        params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {params:,}\n")

    # Test loss
    criterion = MultiTaskLoss()
    targets = torch.randint(0, 4, (2,), device=device)
    damage = torch.rand(2, device=device) * 100
    loss, comps = criterion(outputs, targets, damage)
    print(f"Loss: {loss.item():.4f} (cls: {comps['classification']:.4f}, reg: {comps['regression']:.4f})")
