"""
PyTorch Dataset and DataLoader for CT scan volumes.

Supports loading preprocessed volumes for:
- Segmentation training (image + mask pairs)
- Classification training (image + label)
- Registration training (image pairs)
"""

import os
import logging
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any, Callable
from collections import Counter

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, random_split

from .dicom_loader import load_volume
from .transforms import preprocess_volume

logger = logging.getLogger(__name__)


class CTDataset(Dataset):
    """
    General-purpose CT scan dataset.

    Supports multiple tasks:
    - segmentation: loads (volume, mask) pairs
    - classification: loads (volume, label) pairs
    - registration: loads (moving, fixed) volume pairs

    Directory structure expected:
        data_dir/
        ├── images/          # CT volumes (.nii.gz)
        ├── masks/           # Segmentation masks (.nii.gz) [optional]
        └── labels.csv       # Classification labels [optional]
    """

    def __init__(
        self,
        data_dir: str,
        task: str = "classification",
        target_size: Tuple[int, int, int] = (128, 128, 128),
        transform: Optional[Callable] = None,
        preprocess: bool = True,
        max_samples: Optional[int] = None,
        augment: bool = False,
    ):
        """
        Args:
            data_dir: Root directory containing 'images/' and optionally 'masks/' or 'labels.csv'.
            task: One of 'segmentation', 'classification', 'registration'.
            target_size: Volume dimensions after preprocessing.
            transform: Optional MONAI or custom transform to apply.
            preprocess: If True, apply standard preprocessing pipeline.
            max_samples: Limit number of samples (for debugging).
            augment: If True, apply random 3D augmentations (flips, noise, intensity).
        """
        self.data_dir = Path(data_dir)
        self.task = task
        self.target_size = target_size
        self.transform = transform
        self.preprocess = preprocess
        self.augment = augment

        # Discover image files
        images_dir = self.data_dir / "images"
        if not images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {images_dir}")

        self.image_paths = sorted([
            p for p in images_dir.iterdir()
            if p.suffix in (".nii", ".gz", ".dcm", ".npz", ".npy") or p.is_dir()
        ])

        if max_samples:
            self.image_paths = self.image_paths[:max_samples]

        logger.info(f"Found {len(self.image_paths)} volumes for task '{task}'")

        # Pre-cache labels from .npz metadata for classification
        self._cached_labels = {}
        self._cached_damage = {}
        if task == "classification":
            # Try loading labels from .npz files first (fastest)
            npz_count = 0
            for p in self.image_paths:
                if p.suffix == ".npz":
                    try:
                        data = np.load(str(p), allow_pickle=True)
                        self._cached_labels[str(p)] = int(data["severity_class"])
                        self._cached_damage[str(p)] = float(data["damage_percent"])
                        npz_count += 1
                    except Exception:
                        self._cached_labels[str(p)] = 0
                        self._cached_damage[str(p)] = 0.0
            if npz_count > 0:
                # Log class distribution
                label_counts = Counter(self._cached_labels.values())
                class_names = ["Normal", "Mild", "Moderate", "Severe"]
                dist_str = ", ".join(f"{class_names[k]}={v}" for k, v in sorted(label_counts.items()))
                logger.info(f"Labels from .npz: {dist_str}")
            else:
                # Fallback to labels.csv
                self.labels = {}
                labels_file = self.data_dir / "labels.csv"
                if labels_file.exists():
                    self._load_labels(labels_file)
                else:
                    logger.warning(f"No labels found. Using dummy labels.")

        # Load mask paths if segmentation task
        self.mask_paths = {}
        if task == "segmentation":
            masks_dir = self.data_dir / "masks"
            if masks_dir.exists():
                for p in masks_dir.iterdir():
                    self.mask_paths[p.stem.replace(".nii", "")] = p
            else:
                logger.warning(f"Masks directory not found: {masks_dir}")

    def _load_labels(self, labels_file: Path):
        """Load classification labels from CSV (filename, label)."""
        import csv

        with open(labels_file, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                filename = row.get("filename", row.get("file", ""))
                label = int(row.get("label", row.get("severity", 0)))
                self.labels[filename] = label

        logger.info(f"Loaded {len(self.labels)} labels from {labels_file}")

    def __len__(self) -> int:
        if self.task == "registration":
            return max(0, len(self.image_paths) - 1)  # pairs
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if self.task == "registration":
            return self._get_registration_pair(idx)
        elif self.task == "segmentation":
            return self._get_segmentation_item(idx)
        else:
            return self._get_classification_item(idx)

    def _load_and_preprocess(self, path: Path) -> np.ndarray:
        """Load and preprocess a single volume."""
        # Handle .npz files (already preprocessed by prepare_data.py)
        if path.suffix == ".npz":
            try:
                data = np.load(str(path))
                return data["volume"].astype(np.float32)
            except Exception as e:
                logger.error(f"Failed to load .npz {path}: {e}")
                return np.zeros(self.target_size, dtype=np.float32)

        # Handle .npy files
        if path.suffix == ".npy":
            try:
                return np.load(str(path)).astype(np.float32)
            except Exception as e:
                logger.error(f"Failed to load .npy {path}: {e}")
                return np.zeros(self.target_size, dtype=np.float32)

        # Handle raw NIfTI/DICOM files
        try:
            result = load_volume(str(path), return_metadata=True)
            if isinstance(result, tuple):
                volume, metadata = result
                spacing = metadata.get("spacing")
            else:
                volume = result
                spacing = None
        except Exception as e:
            logger.error(f"Failed to load {path}: {e}")
            # Return zeros as fallback
            return np.zeros(self.target_size, dtype=np.float32)

        if self.preprocess:
            volume = preprocess_volume(
                volume,
                spacing=spacing,
                target_size=self.target_size,
            )

        return volume

    def _augment_volume(self, volume: np.ndarray) -> np.ndarray:
        """Apply random 3D augmentations for training."""
        # Random flips along each axis (50% chance each) — anatomically valid
        if np.random.random() > 0.5:
            volume = np.flip(volume, axis=0).copy()  # axial
        if np.random.random() > 0.5:
            volume = np.flip(volume, axis=1).copy()  # coronal
        if np.random.random() > 0.5:
            volume = np.flip(volume, axis=2).copy()  # sagittal

        # Random intensity shift (brightness) — 30% chance
        if np.random.random() > 0.7:
            shift = np.random.uniform(-0.05, 0.05)
            volume = volume + shift

        # Random intensity scale (contrast) — 30% chance
        if np.random.random() > 0.7:
            scale = np.random.uniform(0.95, 1.05)
            volume = volume * scale

        # Additive Gaussian noise — 30% chance, subtle
        if np.random.random() > 0.7:
            noise = np.random.normal(0, 0.01, volume.shape).astype(np.float32)
            volume = volume + noise

        # Clip to valid range
        volume = np.clip(volume, 0.0, 1.0)

        return volume

    def get_labels(self) -> List[int]:
        """Return list of labels for all samples (for WeightedRandomSampler)."""
        labels = []
        for p in self.image_paths:
            labels.append(self._cached_labels.get(str(p), 0))
        return labels

    def _get_classification_item(self, idx: int) -> Dict[str, Any]:
        """Get a single (volume, label) pair."""
        path = self.image_paths[idx]
        volume = self._load_and_preprocess(path)

        # Get label from cache (populated during __init__)
        label = self._cached_labels.get(str(path), 0)
        severity_pct = self._cached_damage.get(str(path), 0.0)

        # Apply augmentation if enabled
        if self.augment:
            volume = self._augment_volume(volume)

        sample = {"image": volume, "label": label, "severity_pct": severity_pct}

        if self.transform:
            sample = self.transform(sample)
        else:
            sample["image"] = torch.from_numpy(volume).unsqueeze(0)  # Add channel dim
            sample["label"] = torch.tensor(label, dtype=torch.long)
            sample["severity_pct"] = torch.tensor(severity_pct, dtype=torch.float32)

        return sample

    def _get_segmentation_item(self, idx: int) -> Dict[str, Any]:
        """Get a (volume, mask) pair."""
        path = self.image_paths[idx]
        volume = self._load_and_preprocess(path)

        stem = path.stem.replace(".nii", "")
        mask_path = self.mask_paths.get(stem)

        if mask_path is not None:
            try:
                mask = load_volume(str(mask_path))
                from .transforms import resize_volume
                mask = resize_volume(mask, self.target_size)
                mask = (mask > 0.5).astype(np.float32)
            except Exception as e:
                logger.error(f"Failed to load mask {mask_path}: {e}")
                mask = np.zeros(self.target_size, dtype=np.float32)
        else:
            mask = np.zeros(self.target_size, dtype=np.float32)

        sample = {"image": volume, "label": mask}

        if self.transform:
            sample = self.transform(sample)
        else:
            sample["image"] = torch.from_numpy(volume).unsqueeze(0)
            sample["label"] = torch.from_numpy(mask).unsqueeze(0)

        return sample

    def _get_registration_pair(self, idx: int) -> Dict[str, Any]:
        """Get a pair of volumes for registration."""
        moving = self._load_and_preprocess(self.image_paths[idx])
        fixed = self._load_and_preprocess(self.image_paths[idx + 1])

        sample = {"moving": moving, "fixed": fixed}

        if self.transform:
            sample = self.transform(sample)
        else:
            sample["moving"] = torch.from_numpy(moving).unsqueeze(0)
            sample["fixed"] = torch.from_numpy(fixed).unsqueeze(0)

        return sample


class SyntheticCTDataset(Dataset):
    """
    Synthetic CT dataset for development and testing.
    Generates random 3D volumes with realistic value ranges.
    """

    SEVERITY_CLASSES = ["normal", "mild", "moderate", "severe"]

    def __init__(
        self,
        num_samples: int = 100,
        volume_size: Tuple[int, int, int] = (128, 128, 128),
        task: str = "classification",
        num_classes: int = 4,
    ):
        self.num_samples = num_samples
        self.volume_size = volume_size
        self.task = task
        self.num_classes = num_classes

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        torch.manual_seed(idx)  # Reproducible per-sample

        if self.task == "registration":
            moving = torch.randn(1, *self.volume_size) * 0.3 + 0.5
            fixed = moving + torch.randn_like(moving) * 0.05
            return {"moving": moving, "fixed": fixed}

        image = torch.randn(1, *self.volume_size) * 0.3 + 0.5

        if self.task == "segmentation":
            # Create a rough sphere mask
            d, h, w = self.volume_size
            z, y, x = torch.meshgrid(
                torch.linspace(-1, 1, d),
                torch.linspace(-1, 1, h),
                torch.linspace(-1, 1, w),
                indexing="ij",
            )
            mask = ((x**2 + y**2 + z**2) < 0.6).float().unsqueeze(0)
            return {"image": image, "label": mask}

        # Classification
        label = torch.tensor(idx % self.num_classes, dtype=torch.long)
        severity_pct = torch.tensor(float(idx % self.num_classes) / self.num_classes * 100)
        return {"image": image, "label": label, "severity_pct": severity_pct}


def create_dataloaders(
    data_dir: str,
    task: str = "classification",
    batch_size: int = 4,
    num_workers: int = 4,
    val_split: float = 0.2,
    test_split: float = 0.1,
    target_size: Tuple[int, int, int] = (128, 128, 128),
    train_transform: Optional[Callable] = None,
    val_transform: Optional[Callable] = None,
    use_synthetic: bool = False,
    synthetic_samples: int = 100,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train/val/test DataLoaders.

    Args:
        data_dir: Root data directory.
        task: 'classification', 'segmentation', or 'registration'.
        batch_size: Batch size for all loaders.
        num_workers: Number of data loading workers.
        val_split: Fraction for validation.
        test_split: Fraction for test.
        target_size: Volume dimensions.
        train_transform: Training augmentation.
        val_transform: Validation transform.
        use_synthetic: Use synthetic data for development.
        synthetic_samples: Number of synthetic samples.

    Returns:
        (train_loader, val_loader, test_loader)
    """
    if use_synthetic:
        dataset = SyntheticCTDataset(
            num_samples=synthetic_samples,
            volume_size=target_size,
            task=task,
        )
        # Split synthetic dataset
        total = len(dataset)
        test_size = int(total * test_split)
        val_size = int(total * val_split)
        train_size = total - val_size - test_size
        train_ds, val_ds, test_ds = random_split(
            dataset,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42),
        )
        logger.info(f"Synthetic dataset split: train={train_size}, val={val_size}, test={test_size}")
    else:
        data_path = Path(data_dir)

        # Check if data has pre-split directories (train/val/test)
        has_splits = (
            (data_path / "train" / "images").exists()
            and (data_path / "val" / "images").exists()
        )

        if has_splits:
            logger.info(f"Using pre-split directories in {data_dir}")
            train_ds = CTDataset(
                data_dir=str(data_path / "train"),
                task=task,
                target_size=target_size,
                transform=train_transform,
                augment=True,  # Enable augmentation for training
            )
            val_ds = CTDataset(
                data_dir=str(data_path / "val"),
                task=task,
                target_size=target_size,
                transform=val_transform,
            )
            # Test split is optional
            if (data_path / "test" / "images").exists():
                test_ds = CTDataset(
                    data_dir=str(data_path / "test"),
                    task=task,
                    target_size=target_size,
                    transform=val_transform,
                )
            else:
                test_ds = val_ds
            logger.info(f"Pre-split sizes: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")
        else:
            # Single directory — use random_split
            dataset = CTDataset(
                data_dir=data_dir,
                task=task,
                target_size=target_size,
                transform=train_transform,
            )
            total = len(dataset)
            test_size = int(total * test_split)
            val_size = int(total * val_split)
            train_size = total - val_size - test_size
            train_ds, val_ds, test_ds = random_split(
                dataset,
                [train_size, val_size, test_size],
                generator=torch.Generator().manual_seed(42),
            )
            logger.info(f"Random split: train={train_size}, val={val_size}, test={test_size}")

    pin_memory = torch.cuda.is_available()

    # Build class-weighted sampler for classification tasks
    train_sampler = None
    train_shuffle = True
    if task == "classification" and hasattr(train_ds, "get_labels"):
        labels = train_ds.get_labels()
        class_counts = Counter(labels)
        num_samples = len(labels)
        # Inverse frequency weighting
        class_weights = {c: num_samples / count for c, count in class_counts.items()}
        sample_weights = [class_weights[l] for l in labels]
        train_sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=num_samples,
            replacement=True,
        )
        train_shuffle = False  # sampler and shuffle are mutually exclusive
        class_names = {0: "Normal", 1: "Mild", 2: "Moderate", 3: "Severe"}
        weight_str = ", ".join(f"{class_names.get(k, k)}={v:.2f}" for k, v in sorted(class_weights.items()))
        logger.info(f"Class-weighted sampling enabled: {weight_str}")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=train_shuffle,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        persistent_workers=num_workers > 0,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Quick verification with synthetic data
    train_dl, val_dl, test_dl = create_dataloaders(
        data_dir=".",
        task="classification",
        batch_size=2,
        num_workers=0,
        use_synthetic=True,
        synthetic_samples=20,
    )

    batch = next(iter(train_dl))
    print(f"Image batch shape: {batch['image'].shape}")
    print(f"Label batch shape: {batch['label'].shape}")
    print(f"Train batches: {len(train_dl)}, Val: {len(val_dl)}, Test: {len(test_dl)}")
    print("Dataset module loaded successfully.")
