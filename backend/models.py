"""
SQLAlchemy async database models for the CT analysis system.
"""

import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import (
    Column,
    String,
    Float,
    Integer,
    DateTime,
    Text,
    ForeignKey,
    JSON,
    Enum,
    func,
)
from sqlalchemy.orm import DeclarativeBase, relationship, Mapped, mapped_column
from sqlalchemy.ext.asyncio import AsyncAttrs, create_async_engine, async_sessionmaker


class Base(AsyncAttrs, DeclarativeBase):
    """Base class for all database models."""
    pass


class Patient(Base):
    """Patient record."""

    __tablename__ = "patients"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    patient_id: Mapped[str] = mapped_column(String(100), unique=True, index=True)
    patient_name: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    # Relationships
    scans: Mapped[list["CTScan"]] = relationship(back_populates="patient", cascade="all, delete-orphan")


class CTScan(Base):
    """Individual CT scan record."""

    __tablename__ = "ct_scans"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    patient_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("patients.id"), index=True
    )
    file_path: Mapped[str] = mapped_column(String(500))
    file_type: Mapped[str] = mapped_column(String(20))  # 'dicom' or 'nifti'
    study_date: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    scan_metadata: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    uploaded_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    # Relationships
    patient: Mapped["Patient"] = relationship(back_populates="scans")
    result: Mapped[Optional["AnalysisResult"]] = relationship(
        back_populates="scan", uselist=False, cascade="all, delete-orphan"
    )


class AnalysisResult(Base):
    """Analysis result for a CT scan."""

    __tablename__ = "analysis_results"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    scan_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("ct_scans.id"), unique=True, index=True
    )

    # Classification results
    severity: Mapped[int] = mapped_column(Integer)  # 0-3
    severity_label: Mapped[str] = mapped_column(String(20))
    confidence: Mapped[float] = mapped_column(Float)
    damage_percent: Mapped[float] = mapped_column(Float)
    probabilities: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

    # Longitudinal results (optional)
    change_class: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    change_label: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    change_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    baseline_scan_id: Mapped[Optional[str]] = mapped_column(
        String(36), ForeignKey("ct_scans.id"), nullable=True
    )

    # Generated files
    segmentation_path: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    heatmap_path: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    difference_map_path: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)

    # Metadata
    processing_time: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    stage_times: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    # Relationships
    scan: Mapped["CTScan"] = relationship(
        back_populates="result", foreign_keys=[scan_id]
    )


# Database engine factory
def get_engine(database_url: str):
    """Create async database engine."""
    return create_async_engine(database_url, echo=False, pool_size=5, max_overflow=10)


def get_session_factory(engine):
    """Create async session factory."""
    return async_sessionmaker(engine, expire_on_commit=False)
