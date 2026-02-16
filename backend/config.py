"""
Application configuration via environment variables.
"""

import os
from pathlib import Path
from typing import List

from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment / .env file."""

    # Database
    database_url: str = Field(
        default="postgresql+asyncpg://postgres:password@localhost:5432/ct_analysis",
        alias="DATABASE_URL",
    )

    # Model checkpoint paths
    segmentation_model_path: str = Field(
        default="./checkpoints/segmentation/best_model.pt",
        alias="SEGMENTATION_MODEL_PATH",
    )
    registration_model_path: str = Field(
        default="./checkpoints/registration/best_model.pt",
        alias="REGISTRATION_MODEL_PATH",
    )
    classifier_model_path: str = Field(
        default="./checkpoints/classifier/best_model.pt",
        alias="CLASSIFIER_MODEL_PATH",
    )

    # GPU
    cuda_visible_devices: str = Field(default="0", alias="CUDA_VISIBLE_DEVICES")
    mixed_precision: bool = Field(default=True, alias="MIXED_PRECISION")

    # Storage
    upload_dir: str = Field(default="./uploads", alias="UPLOAD_DIR")
    results_dir: str = Field(default="./results", alias="RESULTS_DIR")

    # MLflow
    mlflow_tracking_uri: str = Field(
        default="http://localhost:5000", alias="MLFLOW_TRACKING_URI"
    )
    mlflow_experiment_name: str = Field(
        default="ct-analysis", alias="MLFLOW_EXPERIMENT_NAME"
    )

    # API
    api_host: str = Field(default="0.0.0.0", alias="API_HOST")
    api_port: int = Field(default=8000, alias="API_PORT")
    cors_origins: str = Field(
        default="http://localhost:5173,http://localhost:3000",
        alias="CORS_ORIGINS",
    )

    @property
    def cors_origins_list(self) -> List[str]:
        return [o.strip() for o in self.cors_origins.split(",")]

    def ensure_directories(self):
        """Create upload and results directories if they don't exist."""
        Path(self.upload_dir).mkdir(parents=True, exist_ok=True)
        Path(self.results_dir).mkdir(parents=True, exist_ok=True)

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        populate_by_name = True


settings = Settings()
