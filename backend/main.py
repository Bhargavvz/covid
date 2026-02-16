"""
FastAPI application entry point.

Lifespan: initializes database, loads ML models, creates storage dirs.
Includes CORS middleware and mounts API routers.
"""

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.config import settings
from backend.models import Base, get_engine, get_session_factory
from backend.routers import upload, results

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    """Application lifespan: startup & shutdown hooks."""

    # ── Startup ──
    logger.info("Starting Post-COVID CT Analysis API...")

    # Ensure storage directories exist
    settings.ensure_directories()

    # Initialize database
    engine = get_engine(settings.database_url)
    app.state.engine = engine
    app.state.session_factory = get_session_factory(engine)

    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("Database tables initialized")
    except Exception as e:
        logger.warning(f"Database init skipped (will retry on first request): {e}")

    # Initialize inference pipeline
    try:
        from src.inference.pipeline import InferencePipeline

        pipeline = InferencePipeline(
            seg_model_path=settings.segmentation_model_path,
            reg_model_path=settings.registration_model_path,
            cls_model_path=settings.classifier_model_path,
            use_amp=settings.mixed_precision,
        )
        app.state.pipeline = pipeline
        logger.info("Inference pipeline initialized")
    except Exception as e:
        logger.warning(f"Inference pipeline init failed: {e}")
        app.state.pipeline = None

    # GPU info
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_mem / (1024 ** 3)
        logger.info(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    else:
        logger.info("Running on CPU (no CUDA GPU detected)")

    yield

    # ── Shutdown ──
    logger.info("Shutting down...")
    await engine.dispose()


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""

    app = FastAPI(
        title="Post-COVID CT Scan Analysis API",
        description=(
            "Production-ready deep learning system for analyzing lung CT scans "
            "to detect, quantify, and track Post-COVID pulmonary abnormalities."
        ),
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/api/docs",
        redoc_url="/api/redoc",
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Static files for results (heatmaps, segmentations)
    import os
    os.makedirs(settings.results_dir, exist_ok=True)
    app.mount(
        "/static/results",
        StaticFiles(directory=settings.results_dir),
        name="results",
    )

    # API Routers
    app.include_router(upload.router, prefix="/api", tags=["Upload & Analysis"])
    app.include_router(results.router, prefix="/api", tags=["Results & Visualization"])

    @app.get("/api/health", tags=["Health"])
    async def health_check():
        """Health check endpoint."""
        from backend.schemas import HealthResponse

        pipeline = getattr(app.state, "pipeline", None)
        return HealthResponse(
            status="healthy",
            gpu_available=torch.cuda.is_available(),
            models_loaded={
                "segmentation": pipeline is not None and pipeline._seg_model is not None,
                "registration": pipeline is not None and pipeline._reg_model is not None,
                "classifier": pipeline is not None and pipeline._cls_model is not None,
            },
        )

    return app


# Application instance
app = create_app()
