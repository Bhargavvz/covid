"""
Longitudinal change detection model.

Dual-input architecture that takes a registered scan pair
(baseline + follow-up) and classifies change as:
- Improved
- Stable  
- Worsened
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

from .resnet3d import ResNet3D


class LongitudinalModel(nn.Module):
    """
    Dual-input model for longitudinal change detection.

    Takes registered baseline and follow-up CT scans as input,
    extracts features from each using shared or separate encoders,
    and predicts the change status.

    Outputs:
        - change_logits: (B, 3) for [Improved, Stable, Worsened]
        - change_score: (B, 1) continuous change magnitude
    """

    CHANGE_LABELS = {0: "Improved", 1: "Stable", 2: "Worsened"}

    def __init__(
        self,
        in_channels: int = 1,
        resnet_depth: int = 18,
        feature_dim: int = 256,
        num_change_classes: int = 3,
        shared_encoder: bool = True,
        dropout: float = 0.3,
        include_difference: bool = True,
    ):
        """
        Args:
            in_channels: Input channels per scan.
            resnet_depth: ResNet depth for feature extraction.
            feature_dim: Feature dimension for comparison.
            num_change_classes: Number of change categories.
            shared_encoder: If True, use shared weights for both scans.
            dropout: Dropout probability.
            include_difference: If True, also feed difference map as input.
        """
        super().__init__()
        self.shared_encoder = shared_encoder
        self.include_difference = include_difference

        # Feature extractors
        self.encoder_baseline = ResNet3D(
            in_channels=in_channels,
            num_classes=4,  # placeholder, we use features only
            depth=resnet_depth,
            feature_dim=feature_dim,
        )

        if shared_encoder:
            self.encoder_followup = self.encoder_baseline
        else:
            self.encoder_followup = ResNet3D(
                in_channels=in_channels,
                num_classes=4,
                depth=resnet_depth,
                feature_dim=feature_dim,
            )

        # Difference map encoder (lighter)
        if include_difference:
            self.diff_encoder = nn.Sequential(
                nn.Conv3d(1, 16, 3, stride=2, padding=1),
                nn.BatchNorm3d(16),
                nn.ReLU(inplace=True),
                nn.Conv3d(16, 32, 3, stride=2, padding=1),
                nn.BatchNorm3d(32),
                nn.ReLU(inplace=True),
                nn.Conv3d(32, 64, 3, stride=2, padding=1),
                nn.BatchNorm3d(64),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool3d(1),
                nn.Flatten(),
                nn.Linear(64, feature_dim // 2),
                nn.ReLU(inplace=True),
            )
            comparison_dim = feature_dim * 2 + feature_dim + feature_dim // 2
        else:
            comparison_dim = feature_dim * 2 + feature_dim

        # Comparison network
        self.comparison = nn.Sequential(
            nn.Linear(comparison_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # Change classification head
        self.change_classifier = nn.Linear(feature_dim // 2, num_change_classes)

        # Change magnitude regression head
        self.change_regressor = nn.Sequential(
            nn.Linear(feature_dim // 2, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Tanh(),  # Output in [-1, 1], negative = improved, positive = worsened
        )

    def forward(
        self,
        baseline: torch.Tensor,
        followup: torch.Tensor,
        difference_map: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            baseline: Baseline scan (B, 1, D, H, W).
            followup: Registered follow-up scan (B, 1, D, H, W).
            difference_map: Optional difference map (B, 1, D, H, W).

        Returns:
            Dict with:
                - 'change_logits': (B, 3)
                - 'change_score': (B, 1) in [-1, 1]
                - 'features_baseline': (B, feature_dim)
                - 'features_followup': (B, feature_dim)
        """
        # Extract features
        feat_baseline = self.encoder_baseline.extract_features(baseline)
        feat_followup = self.encoder_followup.extract_features(followup)

        # Feature difference
        feat_diff = feat_followup - feat_baseline

        # Concatenate: [baseline_features, followup_features, feature_difference]
        combined = torch.cat([feat_baseline, feat_followup, feat_diff], dim=1)

        # Add difference map features if available
        if self.include_difference and difference_map is not None:
            diff_feat = self.diff_encoder(difference_map)
            combined = torch.cat([combined, diff_feat], dim=1)
        elif self.include_difference:
            # Compute difference map from inputs
            diff = torch.abs(followup - baseline)
            diff_feat = self.diff_encoder(diff)
            combined = torch.cat([combined, diff_feat], dim=1)

        # Compare
        comparison_feat = self.comparison(combined)

        # Predict change
        change_logits = self.change_classifier(comparison_feat)
        change_score = self.change_regressor(comparison_feat)

        return {
            "change_logits": change_logits,
            "change_score": change_score,
            "features_baseline": feat_baseline,
            "features_followup": feat_followup,
        }

    def predict(self, baseline, followup, difference_map=None) -> Dict[str, torch.Tensor]:
        """Inference prediction with class labels."""
        self.eval()
        with torch.no_grad():
            outputs = self.forward(baseline, followup, difference_map)
            probs = F.softmax(outputs["change_logits"], dim=1)
            change_class = probs.argmax(dim=1)

        return {
            "change_class": change_class,
            "change_label": [self.CHANGE_LABELS[c.item()] for c in change_class],
            "change_probs": probs,
            "change_score": outputs["change_score"].squeeze(-1),
        }


class LongitudinalLoss(nn.Module):
    """Combined loss for longitudinal change detection."""

    def __init__(self, cls_weight: float = 1.0, reg_weight: float = 0.5):
        super().__init__()
        self.cls_weight = cls_weight
        self.reg_weight = reg_weight
        self.ce = nn.CrossEntropyLoss()
        self.mse = nn.MSELoss()

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        change_targets: torch.Tensor,
        score_targets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        cls_loss = self.ce(outputs["change_logits"], change_targets)

        reg_loss = torch.tensor(0.0, device=cls_loss.device)
        if score_targets is not None:
            reg_loss = self.mse(outputs["change_score"].squeeze(-1), score_targets)

        total = self.cls_weight * cls_loss + self.reg_weight * reg_loss

        return total, {
            "classification": cls_loss.item(),
            "regression": reg_loss.item(),
            "total": total.item(),
        }


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LongitudinalModel(resnet_depth=18, feature_dim=256).to(device)

    baseline = torch.randn(2, 1, 64, 64, 64, device=device)
    followup = torch.randn(2, 1, 64, 64, 64, device=device)

    outputs = model(baseline, followup)
    print(f"Change logits: {outputs['change_logits'].shape}")
    print(f"Change score: {outputs['change_score'].shape}")

    preds = model.predict(baseline, followup)
    print(f"Predictions: {preds['change_label']}")
    print(f"Change scores: {preds['change_score']}")

    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,}")
