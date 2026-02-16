"""
Tests for the CT analysis system.
"""

import pytest
import numpy as np
import torch


class TestPreprocessing:
    """Tests for the preprocessing pipeline."""

    def test_apply_lung_window(self):
        from src.preprocessing.transforms import apply_lung_window

        vol = np.array([-1500, -1000, 0, 400, 1000], dtype=np.float32)
        result = apply_lung_window(vol)
        assert result.min() >= 0.0
        assert result.max() <= 1.0
        assert result.dtype == np.float32

    def test_resize_volume(self):
        from src.preprocessing.transforms import resize_volume

        vol = np.random.rand(32, 64, 64).astype(np.float32)
        result = resize_volume(vol, target_size=(16, 32, 32))
        assert result.shape == (16, 32, 32)

    def test_preprocess_volume(self):
        from src.preprocessing.transforms import preprocess_volume

        vol = np.random.uniform(-1000, 400, size=(48, 128, 128)).astype(np.float32)
        result = preprocess_volume(vol, spacing=None, target_size=(64, 64, 64))
        assert result.shape == (64, 64, 64)
        assert 0.0 <= result.min() <= result.max() <= 1.0


class TestSegmentation:
    """Tests for the 3D U-Net segmentation model."""

    def test_unet3d_forward(self):
        from src.segmentation.unet3d import UNet3D

        model = UNet3D(in_channels=1, out_channels=1, use_monai=False)
        x = torch.randn(1, 1, 32, 32, 32)
        out = model(x)
        assert out.shape == (1, 1, 32, 32, 32)

    def test_unet3d_predict(self):
        from src.segmentation.unet3d import UNet3D

        model = UNet3D(in_channels=1, out_channels=1, use_monai=False)
        x = torch.randn(1, 1, 32, 32, 32)
        mask = model.predict(x)
        assert mask.shape == (1, 1, 32, 32, 32)
        assert set(mask.unique().tolist()).issubset({0.0, 1.0})

    def test_dice_score(self):
        from src.segmentation.metrics import dice_score

        pred = torch.ones(1, 1, 8, 8, 8)
        target = torch.ones(1, 1, 8, 8, 8)
        score = dice_score(pred, target, threshold=None)
        assert abs(score.item() - 1.0) < 1e-5

    def test_dice_bce_loss(self):
        from src.segmentation.metrics import DiceBCELoss

        loss_fn = DiceBCELoss()
        pred = torch.randn(2, 1, 16, 16, 16)
        target = (torch.randn(2, 1, 16, 16, 16) > 0).float()
        loss = loss_fn(pred, target)
        assert loss.item() > 0


class TestRegistration:
    """Tests for VoxelMorph registration."""

    def test_voxelmorph_forward(self):
        from src.registration.voxelmorph import VoxelMorph3D

        vol_size = (32, 32, 32)
        model = VoxelMorph3D(volume_size=vol_size)
        moving = torch.randn(1, 1, *vol_size)
        fixed = torch.randn(1, 1, *vol_size)
        moved, flow = model(moving, fixed)
        assert moved.shape == (1, 1, *vol_size)
        assert flow.shape == (1, 3, *vol_size)

    def test_registration_loss(self):
        from src.registration.voxelmorph import RegistrationLoss

        loss_fn = RegistrationLoss(similarity="mse")
        moved = torch.randn(1, 1, 16, 16, 16)
        fixed = torch.randn(1, 1, 16, 16, 16)
        flow = torch.randn(1, 3, 16, 16, 16)
        loss, components = loss_fn(moved, fixed, flow)
        assert loss.item() > 0
        assert "similarity" in components
        assert "regularization" in components

    def test_compute_difference_map(self):
        from src.registration.register import compute_difference_map

        baseline = np.random.rand(32, 32, 32).astype(np.float32)
        followup = baseline + np.random.rand(32, 32, 32).astype(np.float32) * 0.1
        diff = compute_difference_map(baseline, followup)
        assert diff.shape == (32, 32, 32)
        assert diff.min() >= 0
        assert diff.max() <= 1.0


class TestClassification:
    """Tests for 3D ResNet classification."""

    def test_resnet3d_forward(self):
        from src.classification.resnet3d import ResNet3D

        model = ResNet3D(in_channels=1, num_classes=4, depth=18)
        x = torch.randn(2, 1, 32, 32, 32)
        outputs = model(x)
        assert outputs["severity_logits"].shape == (2, 4)
        assert outputs["damage_pct"].shape == (2, 1)

    def test_resnet3d_predict(self):
        from src.classification.resnet3d import ResNet3D

        model = ResNet3D(in_channels=1, num_classes=4, depth=18)
        x = torch.randn(1, 1, 32, 32, 32)
        pred = model.predict(x)
        assert "severity_class" in pred
        assert "severity_label" in pred
        assert "severity_prob" in pred
        assert "damage_pct" in pred

    def test_multi_task_loss(self):
        from src.classification.resnet3d import MultiTaskLoss

        loss_fn = MultiTaskLoss(num_classes=4)
        outputs = {
            "severity_logits": torch.randn(4, 4),
            "damage_pct": torch.rand(4, 1) * 100,
        }
        targets = torch.randint(0, 4, (4,))
        damage = torch.rand(4) * 100
        loss, components = loss_fn(outputs, targets, damage)
        assert loss.item() > 0

    def test_longitudinal_model(self):
        from src.classification.longitudinal import LongitudinalModel

        model = LongitudinalModel(resnet_depth=18, feature_dim=128)
        baseline = torch.randn(2, 1, 32, 32, 32)
        followup = torch.randn(2, 1, 32, 32, 32)
        outputs = model(baseline, followup)
        assert outputs["change_logits"].shape == (2, 3)
        assert outputs["change_score"].shape == (2, 1)

    def test_classification_metrics(self):
        from src.classification.metrics import ClassificationMetrics

        metrics = ClassificationMetrics(num_classes=4)
        for _ in range(5):
            logits = torch.randn(8, 4)
            targets = torch.randint(0, 4, (8,))
            metrics.update(logits, targets)

        results = metrics.compute()
        assert "accuracy" in results
        assert "f1" in results
        assert 0 <= results["accuracy"] <= 1


class TestInference:
    """Tests for the inference pipeline."""

    def test_pipeline_init(self):
        from src.inference.pipeline import InferencePipeline

        pipeline = InferencePipeline(use_amp=False)
        assert pipeline.device is not None
        assert pipeline.VOLUME_SIZE == (128, 128, 128)

    def test_heatmap_generation(self):
        from src.inference.pipeline import InferencePipeline

        vol_slice = np.random.rand(64, 64).astype(np.float32)
        overlay = np.random.rand(64, 64).astype(np.float32)
        heatmap = InferencePipeline.generate_heatmap_slice(vol_slice, overlay)
        assert heatmap.shape == (64, 64, 3)
        assert heatmap.dtype == np.uint8


class TestBackend:
    """Tests for backend schemas."""

    def test_upload_response_schema(self):
        from backend.schemas import UploadResponse

        resp = UploadResponse(
            scan_id="abc-123",
            patient_id="PAT-001",
            severity="Moderate",
            damage_percent=32.5,
            processing_time=12.3,
        )
        assert resp.severity == "Moderate"
        assert resp.damage_percent == 32.5

    def test_analysis_response_schema(self):
        from backend.schemas import AnalysisResponse
        from datetime import datetime

        resp = AnalysisResponse(
            id="res-1",
            scan_id="scan-1",
            severity=2,
            severity_label="Moderate",
            confidence=0.87,
            damage_percent=32.5,
            created_at=datetime.now(),
        )
        assert resp.severity_label == "Moderate"

    def test_health_response(self):
        from backend.schemas import HealthResponse

        resp = HealthResponse(
            status="healthy",
            gpu_available=True,
            models_loaded={"segmentation": True},
        )
        assert resp.status == "healthy"
