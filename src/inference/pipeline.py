"""
End-to-end inference pipeline for Post-COVID CT scan analysis.

Orchestrates:
1. Volume loading & preprocessing
2. Lung segmentation (3D U-Net)
3. Image registration (VoxelMorph, optional follow-up)
4. Severity classification (3D ResNet)
5. Longitudinal change detection (dual-input model)

Supports mixed-precision inference on GPU.
"""

import os
import io
import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from src.preprocessing.dicom_loader import load_volume
from src.preprocessing.transforms import preprocess_volume
from src.segmentation.unet3d import UNet3D
from src.registration.voxelmorph import VoxelMorph3D
from src.registration.register import (
    register_scans,
    compute_difference_map,
    compute_deformation_magnitude,
)
from src.classification.resnet3d import ResNet3D, SEVERITY_LABELS
from src.classification.longitudinal import LongitudinalModel

logger = logging.getLogger(__name__)


class InferencePipeline:
    """
    Production inference pipeline for CT scan analysis.

    Lazily loads model checkpoints on first use. All models are cached
    after loading for rapid subsequent inferences.

    Usage:
        pipeline = InferencePipeline(
            seg_model_path="checkpoints/segmentation/best_model.pt",
            reg_model_path="checkpoints/registration/best_model.pt",
            cls_model_path="checkpoints/classifier/best_model.pt",
        )
        result = pipeline.analyze("path/to/scan.nii.gz")
    """

    VOLUME_SIZE = (128, 128, 128)

    def __init__(
        self,
        seg_model_path: Optional[str] = None,
        reg_model_path: Optional[str] = None,
        cls_model_path: Optional[str] = None,
        long_model_path: Optional[str] = None,
        device: Optional[torch.device] = None,
        use_amp: bool = True,
    ):
        """
        Args:
            seg_model_path: Path to segmentation model checkpoint.
            reg_model_path: Path to registration model checkpoint.
            cls_model_path: Path to classifier model checkpoint.
            long_model_path: Path to longitudinal model checkpoint.
            device: Torch device (auto-detected if None).
            use_amp: Enable automatic mixed precision for inference.
        """
        self.seg_model_path = seg_model_path
        self.reg_model_path = reg_model_path
        self.cls_model_path = cls_model_path
        self.long_model_path = long_model_path
        self.use_amp = use_amp

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        # Lazily loaded models
        self._seg_model: Optional[UNet3D] = None
        self._reg_model: Optional[VoxelMorph3D] = None
        self._cls_model: Optional[ResNet3D] = None
        self._long_model: Optional[LongitudinalModel] = None

        logger.info(f"InferencePipeline initialized (device={self.device}, amp={use_amp})")

    # ── Model Loading ──────────────────────────────────────────────

    def _load_checkpoint(self, path: str) -> Dict[str, Any]:
        """Load a model checkpoint from disk."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        logger.info(f"Loaded checkpoint: {path}")
        return checkpoint

    @property
    def seg_model(self) -> UNet3D:
        """Segmentation model (lazy loaded)."""
        if self._seg_model is None:
            self._seg_model = UNet3D(in_channels=1, out_channels=1).to(self.device)
            if self.seg_model_path and os.path.exists(self.seg_model_path):
                ckpt = self._load_checkpoint(self.seg_model_path)
                state_dict = ckpt.get("model_state_dict", ckpt)
                self._seg_model.load_state_dict(state_dict, strict=False)
            self._seg_model.eval()
        return self._seg_model

    @property
    def reg_model(self) -> VoxelMorph3D:
        """Registration model (lazy loaded)."""
        if self._reg_model is None:
            self._reg_model = VoxelMorph3D(volume_size=self.VOLUME_SIZE).to(self.device)
            if self.reg_model_path and os.path.exists(self.reg_model_path):
                ckpt = self._load_checkpoint(self.reg_model_path)
                state_dict = ckpt.get("model_state_dict", ckpt)
                self._reg_model.load_state_dict(state_dict, strict=False)
            self._reg_model.eval()
        return self._reg_model

    @property
    def cls_model(self) -> ResNet3D:
        """Classification model (lazy loaded)."""
        if self._cls_model is None:
            self._cls_model = ResNet3D(
                in_channels=1, num_classes=4, depth=50
            ).to(self.device)
            if self.cls_model_path and os.path.exists(self.cls_model_path):
                ckpt = self._load_checkpoint(self.cls_model_path)
                state_dict = ckpt.get("model_state_dict", ckpt)
                self._cls_model.load_state_dict(state_dict, strict=False)
            self._cls_model.eval()
        return self._cls_model

    @property
    def long_model(self) -> LongitudinalModel:
        """Longitudinal model (lazy loaded)."""
        if self._long_model is None:
            self._long_model = LongitudinalModel(
                resnet_depth=18, feature_dim=256
            ).to(self.device)
            if self.long_model_path and os.path.exists(self.long_model_path):
                ckpt = self._load_checkpoint(self.long_model_path)
                state_dict = ckpt.get("model_state_dict", ckpt)
                self._long_model.load_state_dict(state_dict, strict=False)
            self._long_model.eval()
        return self._long_model

    # ── Preprocessing ──────────────────────────────────────────────

    def load_and_preprocess(self, scan_path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Load a CT scan and apply full preprocessing.

        Returns:
            volume: Preprocessed volume (D, H, W) in [0, 1].
            metadata: Scan metadata dict.
        """
        volume, metadata = load_volume(scan_path, return_metadata=True)
        spacing = metadata.get("spacing")
        processed = preprocess_volume(
            volume,
            spacing=spacing,
            target_size=self.VOLUME_SIZE,
        )
        logger.info(f"Preprocessed scan: {scan_path} → shape {processed.shape}")
        return processed, metadata

    def _to_tensor(self, volume: np.ndarray) -> torch.Tensor:
        """Convert (D, H, W) numpy array to (1, 1, D, H, W) tensor on device."""
        return (
            torch.from_numpy(volume)
            .float()
            .unsqueeze(0)
            .unsqueeze(0)
            .to(self.device)
        )

    # ── Segmentation ───────────────────────────────────────────────

    def segment(self, volume: np.ndarray) -> np.ndarray:
        """
        Generate binary lung mask from preprocessed volume.

        Args:
            volume: Preprocessed volume (D, H, W).

        Returns:
            Binary lung mask (D, H, W).
        """
        tensor = self._to_tensor(volume)
        with torch.no_grad(), torch.amp.autocast(self.device.type, enabled=self.use_amp):
            mask = self.seg_model.predict(tensor, threshold=0.5)
        return mask.squeeze().cpu().numpy()

    # ── Registration ───────────────────────────────────────────────

    def register(
        self, baseline: np.ndarray, followup: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Register follow-up scan to baseline.

        Returns:
            Dict with 'registered', 'deformation', 'difference_map', 'deformation_magnitude'.
        """
        registered, deformation = register_scans(
            moving=followup,
            fixed=baseline,
            model=self.reg_model,
            device=self.device,
            method="voxelmorph",
        )

        difference = compute_difference_map(baseline, registered, method="absolute")
        magnitude = compute_deformation_magnitude(deformation)

        return {
            "registered": registered,
            "deformation": deformation,
            "difference_map": difference,
            "deformation_magnitude": magnitude,
        }

    # ── Classification ─────────────────────────────────────────────

    def classify(self, volume: np.ndarray) -> Dict[str, Any]:
        """
        Classify severity and estimate lung damage percentage.

        Returns:
            Dict with 'severity', 'severity_label', 'confidence',
                       'damage_percent', 'probabilities'.
        """
        tensor = self._to_tensor(volume)
        with torch.no_grad(), torch.amp.autocast(self.device.type, enabled=self.use_amp):
            prediction = self.cls_model.predict(tensor)

        return {
            "severity": int(prediction["severity_class"].item()),
            "severity_label": prediction["severity_label"][0],
            "confidence": float(prediction["severity_prob"].item()),
            "damage_percent": float(prediction["damage_pct"].item()),
            "probabilities": prediction["probabilities"].cpu().numpy().tolist()[0],
        }

    # ── Longitudinal Analysis ──────────────────────────────────────

    def analyze_change(
        self,
        baseline: np.ndarray,
        followup: np.ndarray,
        difference_map: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Detect longitudinal change between baseline and follow-up.

        Returns:
            Dict with 'change_label', 'change_class', 'change_score'.
        """
        baseline_t = self._to_tensor(baseline)
        followup_t = self._to_tensor(followup)
        diff_t = self._to_tensor(difference_map) if difference_map is not None else None

        with torch.no_grad(), torch.amp.autocast(self.device.type, enabled=self.use_amp):
            prediction = self.long_model.predict(baseline_t, followup_t, diff_t)

        return {
            "change_class": int(prediction["change_class"].item()),
            "change_label": prediction["change_label"][0],
            "change_score": float(prediction["change_score"].item()),
        }

    # ── Full Analysis ──────────────────────────────────────────────

    def analyze(
        self,
        scan_path: str,
        baseline_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run full analysis pipeline on a CT scan.

        Args:
            scan_path: Path to the CT scan (DICOM dir or NIfTI file).
            baseline_path: Optional baseline scan for longitudinal comparison.

        Returns:
            Comprehensive result dict with:
                - severity, damage_percent, confidence
                - segmentation_mask
                - registration results (if baseline provided)
                - longitudinal change (if baseline provided)
                - metadata, timing
        """
        t_start = time.time()
        result: Dict[str, Any] = {"scan_path": scan_path, "stages": {}}

        # 1. Load & preprocess
        t0 = time.time()
        volume, metadata = self.load_and_preprocess(scan_path)
        result["metadata"] = metadata
        result["stages"]["preprocessing"] = time.time() - t0

        # 2. Lung segmentation
        t0 = time.time()
        mask = self.segment(volume)
        result["segmentation_mask"] = mask
        result["stages"]["segmentation"] = time.time() - t0

        # 3. Apply mask to volume (mask the lungs)
        masked_volume = volume * mask

        # 4. Classification
        t0 = time.time()
        classification = self.classify(masked_volume)
        result.update(classification)
        result["stages"]["classification"] = time.time() - t0

        # 5. Registration & longitudinal (if baseline provided)
        if baseline_path is not None:
            t0 = time.time()
            baseline_vol, _ = self.load_and_preprocess(baseline_path)
            reg_result = self.register(baseline_vol, volume)
            result["registration"] = {
                "difference_map": reg_result["difference_map"],
                "deformation_magnitude": reg_result["deformation_magnitude"],
            }
            result["stages"]["registration"] = time.time() - t0

            t0 = time.time()
            change = self.analyze_change(
                baseline_vol,
                reg_result["registered"],
                reg_result["difference_map"],
            )
            result["change"] = change
            result["stages"]["longitudinal"] = time.time() - t0
        else:
            result["change"] = None

        result["total_time"] = time.time() - t_start
        logger.info(
            f"Analysis complete in {result['total_time']:.2f}s — "
            f"Severity: {result['severity_label']}, "
            f"Damage: {result['damage_percent']:.1f}%"
        )

        return result

    # ── Heatmap Generation ─────────────────────────────────────────

    @staticmethod
    def generate_heatmap_slice(
        volume_slice: np.ndarray,
        overlay_slice: np.ndarray,
        alpha: float = 0.4,
    ) -> np.ndarray:
        """
        Generate an RGB heatmap overlay for a single 2D slice.

        Args:
            volume_slice: Base CT slice (H, W) in [0, 1].
            overlay_slice: Overlay data (H, W) in [0, 1].
            alpha: Overlay transparency.

        Returns:
            RGB image (H, W, 3) as uint8.
        """
        # Grayscale base
        base = np.stack([volume_slice] * 3, axis=-1)

        # Color map: blue → green → red
        r = np.clip(overlay_slice * 2, 0, 1)
        g = np.clip(2 * overlay_slice * (1 - overlay_slice) * 4, 0, 1)
        b = np.clip((1 - overlay_slice * 2), 0, 1)
        overlay_rgb = np.stack([r, g, b], axis=-1)

        # Blend
        combined = (1 - alpha) * base + alpha * overlay_rgb
        return (np.clip(combined, 0, 1) * 255).astype(np.uint8)

    def generate_report_slices(
        self,
        result: Dict[str, Any],
        volume: np.ndarray,
        num_slices: int = 5,
    ) -> list:
        """
        Generate representative axial slices with overlays for reporting.

        Returns:
            List of RGB images (H, W, 3) as uint8 arrays.
        """
        depth = volume.shape[0]
        indices = np.linspace(depth * 0.2, depth * 0.8, num_slices, dtype=int)
        slices = []

        mask = result.get("segmentation_mask")
        diff = result.get("registration", {}).get("difference_map")
        overlay = diff if diff is not None else mask

        for idx in indices:
            vol_slice = volume[idx]
            ovl_slice = overlay[idx] if overlay is not None else np.zeros_like(vol_slice)
            rgb = self.generate_heatmap_slice(vol_slice, ovl_slice)
            slices.append(rgb)

        return slices


if __name__ == "__main__":
    # Quick verification with synthetic data (no model weights needed)
    print("Creating InferencePipeline...")
    pipeline = InferencePipeline(use_amp=False)

    # Test preprocessing
    vol = np.random.uniform(-1000, 400, size=(64, 128, 128)).astype(np.float32)
    processed = preprocess_volume(vol, spacing=None)
    print(f"Preprocessed shape: {processed.shape}")

    # Test segmentation
    mask = pipeline.segment(processed)
    print(f"Segmentation mask shape: {mask.shape}, unique: {np.unique(mask)}")

    # Test classification
    cls_result = pipeline.classify(processed)
    print(f"Classification: {cls_result['severity_label']} ({cls_result['confidence']:.2%})")
    print(f"Damage: {cls_result['damage_percent']:.1f}%")

    # Test heatmap generation
    heatmap = pipeline.generate_heatmap_slice(
        processed[64], np.random.rand(128, 128).astype(np.float32)
    )
    print(f"Heatmap shape: {heatmap.shape}, dtype: {heatmap.dtype}")

    print("\nInferencePipeline loaded successfully.")
