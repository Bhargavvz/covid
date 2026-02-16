"""
DICOM and NIfTI volume loading utilities.

Supports loading DICOM series from directories and NIfTI files,
with metadata extraction for spacing, orientation, and patient info.
"""

import os
import logging
from pathlib import Path
from typing import Tuple, Optional, Dict, Any

import numpy as np

logger = logging.getLogger(__name__)


def load_dicom_series(
    dicom_dir: str,
    return_metadata: bool = False,
) -> np.ndarray | Tuple[np.ndarray, Dict[str, Any]]:
    """
    Load a DICOM series from a directory into a 3D NumPy volume.

    Args:
        dicom_dir: Path to directory containing DICOM files.
        return_metadata: If True, also return a metadata dict.

    Returns:
        volume: 3D numpy array (D, H, W) in Hounsfield Units.
        metadata: (optional) dict with spacing, origin, direction, patient info.
    """
    try:
        import pydicom
    except ImportError:
        raise ImportError("pydicom is required: pip install pydicom")

    dicom_dir = Path(dicom_dir)
    if not dicom_dir.is_dir():
        raise FileNotFoundError(f"DICOM directory not found: {dicom_dir}")

    # Collect all DICOM files
    dicom_files = []
    for f in sorted(dicom_dir.iterdir()):
        if f.is_file():
            try:
                ds = pydicom.dcmread(str(f), stop_before_pixels=True)
                if hasattr(ds, "InstanceNumber"):
                    dicom_files.append((f, ds.InstanceNumber))
                else:
                    dicom_files.append((f, 0))
            except Exception:
                continue

    if not dicom_files:
        raise ValueError(f"No valid DICOM files found in {dicom_dir}")

    # Sort by instance number
    dicom_files.sort(key=lambda x: x[1])
    logger.info(f"Found {len(dicom_files)} DICOM slices in {dicom_dir}")

    # Load first slice for metadata
    first_ds = pydicom.dcmread(str(dicom_files[0][0]))
    rows = int(first_ds.Rows)
    cols = int(first_ds.Columns)

    # Build 3D volume
    volume = np.zeros((len(dicom_files), rows, cols), dtype=np.float32)

    for i, (filepath, _) in enumerate(dicom_files):
        ds = pydicom.dcmread(str(filepath))
        pixel_array = ds.pixel_array.astype(np.float32)

        # Apply rescale slope/intercept for HU values
        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        volume[i] = pixel_array * slope + intercept

    logger.info(f"Loaded volume shape: {volume.shape}")

    if return_metadata:
        pixel_spacing = [float(s) for s in getattr(first_ds, "PixelSpacing", [1.0, 1.0])]
        slice_thickness = float(getattr(first_ds, "SliceThickness", 1.0))
        metadata = {
            "spacing": [slice_thickness, pixel_spacing[0], pixel_spacing[1]],
            "origin": [float(x) for x in getattr(first_ds, "ImagePositionPatient", [0, 0, 0])],
            "patient_id": str(getattr(first_ds, "PatientID", "unknown")),
            "patient_name": str(getattr(first_ds, "PatientName", "unknown")),
            "study_date": str(getattr(first_ds, "StudyDate", "unknown")),
            "modality": str(getattr(first_ds, "Modality", "CT")),
            "rows": rows,
            "cols": cols,
            "num_slices": len(dicom_files),
        }
        return volume, metadata

    return volume


def load_nifti(
    filepath: str,
    return_metadata: bool = False,
) -> np.ndarray | Tuple[np.ndarray, Dict[str, Any]]:
    """
    Load a NIfTI file (.nii or .nii.gz) into a 3D NumPy volume.

    Args:
        filepath: Path to NIfTI file.
        return_metadata: If True, also return metadata dict.

    Returns:
        volume: 3D numpy array.
        metadata: (optional) dict with spacing, affine, shape.
    """
    try:
        import nibabel as nib
    except ImportError:
        raise ImportError("nibabel is required: pip install nibabel")

    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"NIfTI file not found: {filepath}")

    img = nib.load(str(filepath))
    volume = np.asarray(img.dataobj, dtype=np.float32)

    # Ensure 3D (handle 4D with single timepoint)
    if volume.ndim == 4:
        volume = volume[:, :, :, 0]
        logger.warning(f"4D NIfTI detected, using first volume. Shape: {volume.shape}")

    # Reorient to (D, H, W) if needed — NIfTI is typically (W, H, D)
    if volume.ndim == 3:
        volume = np.transpose(volume, (2, 1, 0))

    logger.info(f"Loaded NIfTI volume shape: {volume.shape}")

    if return_metadata:
        affine = img.affine
        spacing = np.abs(np.diag(affine[:3, :3])).tolist()
        metadata = {
            "spacing": spacing,
            "affine": affine.tolist(),
            "shape": list(volume.shape),
            "dtype": str(volume.dtype),
        }
        return volume, metadata

    return volume


def convert_dicom_to_nifti(
    dicom_dir: str,
    output_path: str,
) -> str:
    """
    Convert a DICOM series to NIfTI format.

    Args:
        dicom_dir: Path to directory with DICOM files.
        output_path: Path for output .nii.gz file.

    Returns:
        output_path: Path to the created NIfTI file.
    """
    try:
        import SimpleITK as sitk
    except ImportError:
        raise ImportError("SimpleITK is required: pip install SimpleITK")

    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(str(dicom_dir))
    if not dicom_names:
        raise ValueError(f"No DICOM series found in {dicom_dir}")

    reader.SetFileNames(dicom_names)
    image = reader.Execute()

    output_path = str(output_path)
    if not output_path.endswith((".nii", ".nii.gz")):
        output_path += ".nii.gz"

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    sitk.WriteImage(image, output_path)
    logger.info(f"Converted DICOM to NIfTI: {output_path}")
    return output_path


def load_volume(
    path: str,
    return_metadata: bool = False,
) -> np.ndarray | Tuple[np.ndarray, Dict[str, Any]]:
    """
    Auto-detect and load a CT volume from DICOM directory or NIfTI file.

    Args:
        path: Path to DICOM directory or NIfTI file.
        return_metadata: If True, also return metadata.

    Returns:
        volume: 3D numpy array.
        metadata: (optional) dict with spacing and other info.
    """
    path = Path(path)

    if path.is_dir():
        return load_dicom_series(str(path), return_metadata=return_metadata)
    elif path.suffix in (".nii", ".gz"):
        return load_nifti(str(path), return_metadata=return_metadata)
    else:
        raise ValueError(f"Unsupported format. Expected DICOM dir or .nii/.nii.gz: {path}")


if __name__ == "__main__":
    # Quick verification with synthetic data
    import tempfile

    print("DICOM/NIfTI loader module loaded successfully.")
    print("Functions: load_dicom_series, load_nifti, convert_dicom_to_nifti, load_volume")
