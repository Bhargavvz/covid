"""
Dataset Preparation Script for Post-COVID CT Scan Analysis
===========================================================

Downloads, extracts, and preprocesses:
  1. MosMedData (COVID19_1110) — NIfTI lung CTs with severity labels
  2. COVID-CTset — DICOM/TIFF CT images (optional)

Usage:
    python scripts/prepare_data.py --dataset mosmed
    python scripts/prepare_data.py --dataset covidctset
    python scripts/prepare_data.py --dataset all
    python scripts/prepare_data.py --preprocess-only   # skip download, just preprocess
"""

import os
import sys
import json
import shutil
import hashlib
import logging
import argparse
import subprocess
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Configuration ───────────────────────────────────────────────────────────
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

MOSMED_URL = (
    "https://storage.yandexcloud.net/covid19.1110/prod/COVID19_1110.7z"
    "?X-Amz-Algorithm=AWS4-HMAC-SHA256"
    "&X-Amz-Credential=J0yLHxYnLMK5SrRghejm%2F20260216%2Fru-central1%2Fs3%2Faws4_request"
    "&X-Amz-Date=20260216T121036Z"
    "&X-Amz-Expires=86400"
    "&X-Amz-SignedHeaders=host"
    "&X-Amz-Signature=82a7a08079e50d539c39374ac349ff87d11cf34ca87d5e59f78830a7a6befd7b"
)

COVIDCTSET_DRIVE_FOLDER = "1xdk-mCkxCDNwsMAk2SGv203rY1mrbnPB"

# MosMedData severity mapping:
#   CT-0  → Normal   (class 0)
#   CT-1  → Mild     (class 1)
#   CT-2  → Moderate (class 2)
#   CT-3  → Severe   (class 3)
#   CT-4  → Severe   (class 3)  ← merged with CT-3
MOSMED_SEVERITY_MAP = {
    "CT-0": {"class": 0, "label": "Normal",   "damage_range": (0, 5)},
    "CT-1": {"class": 1, "label": "Mild",     "damage_range": (5, 25)},
    "CT-2": {"class": 2, "label": "Moderate", "damage_range": (25, 50)},
    "CT-3": {"class": 3, "label": "Severe",   "damage_range": (50, 75)},
    "CT-4": {"class": 3, "label": "Severe",   "damage_range": (75, 100)},
}

TARGET_VOLUME_SIZE = (128, 128, 128)
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15


# ── Download Utilities ──────────────────────────────────────────────────────

def download_file(url: str, output_path: Path, description: str = "file") -> bool:
    """Download a file with progress display."""
    import urllib.request

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        log.info(f"Already downloaded: {output_path.name}")
        return True

    log.info(f"Downloading {description}...")
    log.info(f"  URL: {url[:80]}...")
    log.info(f"  To:  {output_path}")

    try:
        # Try with urllib first
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})

        with urllib.request.urlopen(req, timeout=600) as response:
            total = int(response.headers.get("Content-Length", 0))
            downloaded = 0
            chunk_size = 8192 * 16  # 128KB chunks

            with open(output_path, "wb") as f:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)

                    if total > 0:
                        pct = (downloaded / total) * 100
                        mb_done = downloaded / (1024 * 1024)
                        mb_total = total / (1024 * 1024)
                        print(
                            f"\r  Progress: {pct:.1f}% ({mb_done:.1f}/{mb_total:.1f} MB)",
                            end="", flush=True,
                        )

        print()  # newline after progress
        log.info(f"Download complete: {output_path.name}")
        return True

    except Exception as e:
        log.error(f"urllib download failed: {e}")
        log.info("Attempting download with curl...")

        # Fallback to curl
        try:
            result = subprocess.run(
                ["curl", "-L", "-o", str(output_path), "--progress-bar", url],
                check=True,
                capture_output=False,
            )
            log.info(f"Download complete via curl: {output_path.name}")
            return True
        except (subprocess.CalledProcessError, FileNotFoundError) as e2:
            log.error(f"curl download also failed: {e2}")

            # Cleanup partial download
            if output_path.exists():
                output_path.unlink()

            return False


def extract_7z(archive_path: Path, output_dir: Path) -> bool:
    """Extract a .7z archive."""
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"Extracting {archive_path.name} → {output_dir}")

    # Try py7zr (pure Python)
    try:
        import py7zr

        with py7zr.SevenZipFile(archive_path, mode="r") as z:
            z.extractall(path=output_dir)

        log.info("Extraction complete (py7zr)")
        return True
    except ImportError:
        log.warning("py7zr not installed. Trying 7z command...")
    except Exception as e:
        log.error(f"py7zr extraction failed: {e}")

    # Try system 7z
    try:
        result = subprocess.run(
            ["7z", "x", str(archive_path), f"-o{output_dir}", "-y"],
            check=True,
            capture_output=True,
            text=True,
        )
        log.info("Extraction complete (7z)")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    # Try 7-Zip Windows path
    seven_zip_paths = [
        r"C:\Program Files\7-Zip\7z.exe",
        r"C:\Program Files (x86)\7-Zip\7z.exe",
    ]
    for sz_path in seven_zip_paths:
        if os.path.exists(sz_path):
            try:
                subprocess.run(
                    [sz_path, "x", str(archive_path), f"-o{output_dir}", "-y"],
                    check=True,
                    capture_output=True,
                )
                log.info("Extraction complete (7-Zip)")
                return True
            except subprocess.CalledProcessError:
                continue

    log.error(
        "Cannot extract .7z file. Install one of:\n"
        "  pip install py7zr\n"
        "  or install 7-Zip from https://www.7-zip.org/"
    )
    return False


def download_from_gdrive(folder_id: str, output_dir: Path) -> bool:
    """Download files from Google Drive folder using gdown."""
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        import gdown
    except ImportError:
        log.error("gdown not installed. Run: pip install gdown")
        return False

    log.info(f"Downloading Google Drive folder {folder_id}...")
    log.info(f"  To: {output_dir}")

    try:
        url = f"https://drive.google.com/drive/folders/{folder_id}"
        gdown.download_folder(url, output=str(output_dir), quiet=False)
        log.info("Google Drive download complete")
        return True
    except Exception as e:
        log.error(f"Google Drive download failed: {e}")
        log.info(
            "Manual download:\n"
            f"  1. Go to: https://drive.google.com/drive/folders/{folder_id}\n"
            f"  2. Download all files\n"
            f"  3. Place them in: {output_dir}"
        )
        return False


# ── MosMedData Processing ───────────────────────────────────────────────────

def organize_mosmed(raw_mosmed_dir: Path) -> Dict[str, List[Path]]:
    """
    Organize MosMedData files by severity category.
    MosMedData structure: files named like 'study_XXXX.nii.gz'
    organized in folders: CT-0/, CT-1/, CT-2/, CT-3/, CT-4/
    """
    categories = {}
    nii_files = list(raw_mosmed_dir.rglob("*.nii.gz")) + list(raw_mosmed_dir.rglob("*.nii"))

    if not nii_files:
        log.error(f"No NIfTI files found in {raw_mosmed_dir}")
        return categories

    log.info(f"Found {len(nii_files)} NIfTI files")

    for fpath in sorted(nii_files):
        # Determine category from parent directory name
        category = None
        for part in fpath.parts:
            if part.startswith("CT-") and len(part) == 4:
                category = part
                break

        if category is None:
            # Try to infer from filename
            fname = fpath.stem.replace(".nii", "")
            for cat in ["CT-0", "CT-1", "CT-2", "CT-3", "CT-4"]:
                if cat.lower().replace("-", "") in fname.lower().replace("-", ""):
                    category = cat
                    break

        if category is None:
            category = "unknown"

        if category not in categories:
            categories[category] = []
        categories[category].append(fpath)

    # Report
    log.info("MosMedData distribution:")
    for cat in sorted(categories.keys()):
        info = MOSMED_SEVERITY_MAP.get(cat, {"label": "Unknown", "class": -1})
        log.info(f"  {cat} ({info['label']}): {len(categories[cat])} scans")

    return categories


def preprocess_volume(
    nii_path: Path,
    target_size: Tuple[int, int, int] = TARGET_VOLUME_SIZE,
) -> Optional[np.ndarray]:
    """Load and preprocess a single NIfTI volume."""
    try:
        import nibabel as nib
        from src.preprocessing.transforms import preprocess_volume as prep_vol

        # Load NIfTI
        img = nib.load(str(nii_path))
        volume = img.get_fdata().astype(np.float32)

        # Get spacing if available
        spacing = None
        if hasattr(img.header, "get_zooms"):
            spacing = img.header.get_zooms()[:3]

        # Apply preprocessing pipeline
        processed = prep_vol(volume, spacing=spacing, target_size=target_size)

        return processed

    except Exception as e:
        log.warning(f"Failed to preprocess {nii_path.name}: {e}")
        return None


def create_dataset_splits(
    categories: Dict[str, List[Path]],
    output_dir: Path,
    target_size: Tuple[int, int, int] = TARGET_VOLUME_SIZE,
):
    """
    Preprocess all volumes and split into train/val/test sets.
    Each sample is saved as a .npz file with:
      - 'volume': preprocessed 3D array (D, H, W)
      - 'severity_class': int (0-3)
      - 'severity_label': str
      - 'damage_percent': float (estimated from category range)
      - 'source_file': str
    """
    import random

    random.seed(42)

    # Collect all samples
    all_samples = []
    for cat, files in categories.items():
        if cat == "unknown" or cat not in MOSMED_SEVERITY_MAP:
            log.warning(f"Skipping {len(files)} files from category '{cat}'")
            continue

        info = MOSMED_SEVERITY_MAP[cat]
        for fpath in files:
            all_samples.append({
                "path": fpath,
                "severity_class": info["class"],
                "severity_label": info["label"],
                "damage_range": info["damage_range"],
                "category": cat,
            })

    random.shuffle(all_samples)

    n_total = len(all_samples)
    n_train = int(n_total * TRAIN_SPLIT)
    n_val = int(n_total * VAL_SPLIT)

    splits = {
        "train": all_samples[:n_train],
        "val": all_samples[n_train : n_train + n_val],
        "test": all_samples[n_train + n_val :],
    }

    log.info(f"\nDataset splits — Total: {n_total}")
    for split_name, samples in splits.items():
        log.info(f"  {split_name}: {len(samples)} samples")

    # Process each split
    metadata = {"splits": {}, "target_size": list(target_size), "num_classes": 4}

    for split_name, samples in splits.items():
        split_dir = output_dir / split_name
        images_dir = split_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)

        split_meta = []
        success = 0
        failed = 0

        for i, sample in enumerate(samples):
            log.info(
                f"  [{split_name}] Processing {i + 1}/{len(samples)}: "
                f"{sample['path'].name} ({sample['severity_label']})"
            )

            volume = preprocess_volume(sample["path"], target_size)

            if volume is None:
                failed += 1
                continue

            # Generate damage percentage within category range
            low, high = sample["damage_range"]
            damage_pct = random.uniform(low, high)

            # Save as .npz
            sample_id = f"{split_name}_{success:04d}"
            npz_path = images_dir / f"{sample_id}.npz"

            np.savez_compressed(
                npz_path,
                volume=volume,
                severity_class=sample["severity_class"],
                severity_label=sample["severity_label"],
                damage_percent=damage_pct,
                source_file=sample["path"].name,
            )

            split_meta.append({
                "id": sample_id,
                "file": str(npz_path.relative_to(output_dir)),
                "severity_class": sample["severity_class"],
                "severity_label": sample["severity_label"],
                "damage_percent": round(damage_pct, 2),
                "source": sample["path"].name,
                "category": sample["category"],
            })

            success += 1

        log.info(f"  [{split_name}] Done: {success} success, {failed} failed")
        metadata["splits"][split_name] = {
            "count": success,
            "samples": split_meta,
        }

    # Save metadata
    meta_path = output_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    log.info(f"\nMetadata saved to {meta_path}")

    # Print class distribution per split
    log.info("\nClass distribution:")
    labels = ["Normal", "Mild", "Moderate", "Severe"]
    for split_name in ["train", "val", "test"]:
        counts = [0] * 4
        for s in metadata["splits"][split_name]["samples"]:
            counts[s["severity_class"]] += 1
        dist = " | ".join(f"{labels[i]}: {counts[i]}" for i in range(4))
        log.info(f"  {split_name}: {dist}")

    return metadata


# ── COVID-CTset Processing ──────────────────────────────────────────────────

def organize_covidctset(raw_dir: Path) -> Dict[str, List[Path]]:
    """Organize COVID-CTset files (TIFF/PNG format)."""
    categories = {"COVID": [], "Normal": []}

    for subdir in raw_dir.iterdir():
        if not subdir.is_dir():
            continue

        label = "COVID" if "covid" in subdir.name.lower() else "Normal"
        image_files = (
            list(subdir.rglob("*.tif"))
            + list(subdir.rglob("*.tiff"))
            + list(subdir.rglob("*.png"))
        )
        categories[label].extend(image_files)

    for label, files in categories.items():
        log.info(f"  COVID-CTset {label}: {len(files)} images")

    return categories


# ── Main Pipeline ───────────────────────────────────────────────────────────

def prepare_mosmed(skip_download: bool = False):
    """Full MosMedData preparation pipeline."""
    log.info("=" * 60)
    log.info("  MosMedData (COVID19_1110) Preparation")
    log.info("=" * 60)

    mosmed_raw = RAW_DIR / "mosmed"
    archive_path = RAW_DIR / "COVID19_1110.7z"

    # Step 1: Download
    if not skip_download:
        if not download_file(MOSMED_URL, archive_path, "MosMedData (COVID19_1110.7z)"):
            log.error("Download failed. Exiting.")
            return False

        # Step 2: Extract
        if not mosmed_raw.exists() or not any(mosmed_raw.rglob("*.nii*")):
            if not extract_7z(archive_path, mosmed_raw):
                log.error("Extraction failed. Exiting.")
                return False
        else:
            log.info("Already extracted, skipping.")
    else:
        if not mosmed_raw.exists():
            log.error(f"Raw data not found at {mosmed_raw}. Download first.")
            return False

    # Step 3: Organize by category
    categories = organize_mosmed(mosmed_raw)
    if not categories:
        log.error("No data found to process.")
        return False

    # Step 4: Preprocess and split
    mosmed_processed = PROCESSED_DIR / "mosmed"
    create_dataset_splits(categories, mosmed_processed)

    log.info("\n✅ MosMedData preparation complete!")
    log.info(f"   Processed data: {mosmed_processed}")
    return True


def prepare_covidctset(skip_download: bool = False):
    """Full COVID-CTset preparation pipeline."""
    log.info("=" * 60)
    log.info("  COVID-CTset Preparation")
    log.info("=" * 60)

    ctset_raw = RAW_DIR / "covidctset"

    # Step 1: Download from Google Drive
    if not skip_download:
        if not download_from_gdrive(COVIDCTSET_DRIVE_FOLDER, ctset_raw):
            log.warning("Auto-download failed. See instructions above for manual download.")
            return False
    else:
        if not ctset_raw.exists():
            log.error(f"Raw data not found at {ctset_raw}. Download first.")
            return False

    # Step 2: Organize
    categories = organize_covidctset(ctset_raw)

    log.info("\n✅ COVID-CTset preparation complete!")
    log.info(f"   Raw data: {ctset_raw}")
    return True


def print_summary():
    """Print a summary of available datasets."""
    log.info("\n" + "=" * 60)
    log.info("  Dataset Summary")
    log.info("=" * 60)

    # Check raw data
    if RAW_DIR.exists():
        for subdir in sorted(RAW_DIR.iterdir()):
            if subdir.is_dir():
                n_files = sum(1 for _ in subdir.rglob("*") if _.is_file())
                log.info(f"  Raw: {subdir.name}/ → {n_files} files")

    # Check processed data
    if PROCESSED_DIR.exists():
        for dataset_dir in sorted(PROCESSED_DIR.iterdir()):
            if dataset_dir.is_dir():
                meta_path = dataset_dir / "metadata.json"
                if meta_path.exists():
                    with open(meta_path) as f:
                        meta = json.load(f)
                    for split, info in meta["splits"].items():
                        log.info(f"  Processed: {dataset_dir.name}/{split} → {info['count']} samples")
                else:
                    n_files = sum(1 for _ in dataset_dir.rglob("*.npz"))
                    log.info(f"  Processed: {dataset_dir.name}/ → {n_files} .npz files")
    else:
        log.info("  No processed data found. Run with --preprocess-only or --dataset mosmed")

    # Print next steps
    log.info("\nNext steps:")
    log.info("  1. Train segmentation:")
    log.info("     python -m src.segmentation.train_segmentation \\")
    log.info(f"       --data_dir {PROCESSED_DIR / 'mosmed'} --epochs 100")
    log.info("  2. Train registration:")
    log.info("     python -m src.registration.train_registration \\")
    log.info(f"       --data_dir {PROCESSED_DIR / 'mosmed'} --epochs 100")
    log.info("  3. Train classifier:")
    log.info("     python -m src.classification.train_classifier \\")
    log.info(f"       --data_dir {PROCESSED_DIR / 'mosmed'} --epochs 200")


# ── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Download and prepare datasets for Post-COVID CT Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/prepare_data.py --dataset mosmed         # Download + preprocess MosMedData
  python scripts/prepare_data.py --dataset covidctset     # Download COVID-CTset
  python scripts/prepare_data.py --dataset all            # Download everything
  python scripts/prepare_data.py --preprocess-only        # Preprocess already-downloaded data
  python scripts/prepare_data.py --summary                # Show dataset status
        """,
    )
    parser.add_argument(
        "--dataset",
        choices=["mosmed", "covidctset", "all"],
        default="mosmed",
        help="Which dataset to download and prepare (default: mosmed)",
    )
    parser.add_argument(
        "--preprocess-only",
        action="store_true",
        help="Skip download, only preprocess existing raw data",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print summary of available datasets and exit",
    )
    parser.add_argument(
        "--target-size",
        type=int,
        nargs=3,
        default=[128, 128, 128],
        metavar=("D", "H", "W"),
        help="Target volume size (default: 128 128 128)",
    )

    args = parser.parse_args()

    global TARGET_VOLUME_SIZE
    TARGET_VOLUME_SIZE = tuple(args.target_size)

    if args.summary:
        print_summary()
        return

    log.info("Post-COVID CT Scan Analysis — Dataset Preparation")
    log.info(f"  Project root:  {PROJECT_ROOT}")
    log.info(f"  Raw data dir:  {RAW_DIR}")
    log.info(f"  Processed dir: {PROCESSED_DIR}")
    log.info(f"  Target size:   {TARGET_VOLUME_SIZE}")
    log.info("")

    # Create directories
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    success = True

    if args.dataset in ("mosmed", "all"):
        success &= prepare_mosmed(skip_download=args.preprocess_only)

    if args.dataset in ("covidctset", "all"):
        success &= prepare_covidctset(skip_download=args.preprocess_only)

    print_summary()

    if success:
        log.info("\n🎉 Dataset preparation completed successfully!")
    else:
        log.warning("\n⚠️  Some steps failed. Check the logs above.")


if __name__ == "__main__":
    main()
