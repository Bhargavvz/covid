# Post-COVID CT Scan Analysis System

A production-ready deep learning system for analyzing lung CT scans to detect, quantify, and track Post-COVID pulmonary abnormalities using image registration and CNN-based models.

## Architecture

```
CT Scan (DICOM/NIfTI)
        │
        ▼
┌─────────────────┐
│   Preprocessing  │  Resample → Window → Normalize → Resize
└────────┬────────┘
         ▼
┌─────────────────┐
│  Lung Segmentation│  3D U-Net (MONAI) → Binary Lung Mask
└────────┬────────┘
         ▼
┌─────────────────┐
│ Image Registration│  VoxelMorph 3D → Deformation Field + Difference Map
└────────┬────────┘
         ▼
┌─────────────────┐
│ CNN Classifier   │  3D ResNet → Severity + % Damage + Longitudinal Change
└────────┬────────┘
         ▼
┌─────────────────┐
│  FastAPI Backend │  REST API → Upload, Analyze, Retrieve Results
└────────┬────────┘
         ▼
┌─────────────────┐
│ React Dashboard  │  CT Viewer + Heatmaps + Charts + Patient History
└─────────────────┘
```

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Backend
```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

### 3. Run Frontend
```bash
cd frontend && npm install && npm run dev
```

### 4. Docker (Full Stack)
```bash
cd deploy && docker-compose up -d
```

## Training

### Segmentation
```bash
python -m src.segmentation.train_segmentation --data_dir ./data --epochs 100
```

### Registration
```bash
python -m src.registration.train_registration --data_dir ./data --epochs 100
```

### Classifier
```bash
python -m src.classification.train_classifier --data_dir ./data --epochs 200
```

## Project Structure

```
├── src/                    # ML Pipeline
│   ├── preprocessing/      # DICOM loading, transforms, datasets
│   ├── segmentation/       # 3D U-Net lung segmentation
│   ├── registration/       # VoxelMorph image registration
│   ├── classification/     # 3D ResNet severity classification
│   └── inference/          # End-to-end inference pipeline
├── backend/                # FastAPI REST API
├── frontend/               # React Dashboard
├── deploy/                 # Docker, Nginx, Triton configs
├── mlops/                  # DVC, MLflow, CI/CD
├── tests/                  # Unit & integration tests
└── checkpoints/            # Saved model weights
```

## Hardware Requirements

- **Training**: NVIDIA H200 (141GB VRAM) recommended
- **Inference**: Any NVIDIA GPU with 8GB+ VRAM
- **Mixed Precision**: FP16/BF16 enabled by default

## Datasets

| Dataset | Purpose |
|---------|---------|
| MosMedData | COVID severity classification |
| COVID-CTset | Large COVID CT dataset |
| SARS-CoV-2 CT-scan | Binary COVID detection |
| LIDC-IDRI | Lung structure pretraining |
| NSCLC Radiomics (TCIA) | Longitudinal registration |
