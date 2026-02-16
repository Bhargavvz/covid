"""
Upload & analysis API routes.

POST /api/upload — Upload CT scan (DICOM/NIfTI) and run analysis
GET  /api/patients — List all patients
GET  /api/patients/{patient_id}/scans — Get patient scan history
GET  /api/patients/{patient_id}/damage-history — Longitudinal damage timeline
"""

import os
import uuid
import shutil
import logging
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Request, BackgroundTasks
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from backend.config import settings
from backend.models import Patient, CTScan, AnalysisResult
from backend.schemas import (
    UploadResponse,
    PatientResponse,
    PatientDetailResponse,
    ScanResponse,
    DamageHistoryResponse,
    DamageHistoryPoint,
)

logger = logging.getLogger(__name__)
router = APIRouter()


def _get_session(request: Request):
    """Get async session from app state."""
    return request.app.state.session_factory()


def _get_pipeline(request: Request):
    """Get inference pipeline from app state."""
    pipeline = getattr(request.app.state, "pipeline", None)
    if pipeline is None:
        raise HTTPException(503, "Inference pipeline not available")
    return pipeline


def _detect_file_type(filename: str) -> str:
    """Detect DICOM or NIfTI from filename."""
    name = filename.lower()
    if name.endswith((".nii", ".nii.gz")):
        return "nifti"
    elif name.endswith(".dcm"):
        return "dicom"
    else:
        return "dicom"  # Default assumption for DICOM files without extension


@router.post("/upload", response_model=UploadResponse)
async def upload_and_analyze(
    request: Request,
    file: UploadFile = File(..., description="CT scan file (DICOM or NIfTI)"),
    patient_id: str = Form(..., description="Patient identifier"),
    patient_name: Optional[str] = Form(None, description="Patient name"),
    baseline_scan_id: Optional[str] = Form(None, description="Baseline scan ID for comparison"),
):
    """
    Upload a CT scan and run the full analysis pipeline.

    1. Saves the uploaded file
    2. Runs preprocessing → segmentation → classification
    3. If baseline_scan_id provided, also runs registration + longitudinal analysis
    4. Stores results in database
    """
    pipeline = _get_pipeline(request)

    # Save uploaded file
    file_type = _detect_file_type(file.filename or "scan.nii.gz")
    scan_id = str(uuid.uuid4())
    upload_subdir = Path(settings.upload_dir) / scan_id
    upload_subdir.mkdir(parents=True, exist_ok=True)
    file_path = upload_subdir / (file.filename or f"scan.{file_type}")

    try:
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        logger.info(f"Saved upload: {file_path} ({len(content)} bytes)")
    except Exception as e:
        raise HTTPException(500, f"Failed to save file: {e}")

    # Get baseline path if specified
    baseline_path = None
    if baseline_scan_id:
        async with _get_session(request) as session:
            baseline_scan = await session.get(CTScan, baseline_scan_id)
            if baseline_scan:
                baseline_path = baseline_scan.file_path

    # Run analysis
    try:
        result = pipeline.analyze(
            scan_path=str(file_path),
            baseline_path=baseline_path,
        )
    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        raise HTTPException(500, f"Analysis failed: {e}")

    # Save result artifacts
    results_subdir = Path(settings.results_dir) / scan_id
    results_subdir.mkdir(parents=True, exist_ok=True)

    seg_path = None
    heatmap_path = None
    diff_path = None

    try:
        # Save segmentation mask
        mask = result.get("segmentation_mask")
        if mask is not None:
            seg_file = results_subdir / "segmentation.npy"
            np.save(str(seg_file), mask)
            seg_path = str(seg_file)

        # Save heatmap slices
        if "registration" in result and result["registration"]:
            diff_map = result["registration"].get("difference_map")
            if diff_map is not None:
                diff_file = results_subdir / "difference_map.npy"
                np.save(str(diff_file), diff_map)
                diff_path = str(diff_file)

        # Generate and save heatmap image
        volume, _ = pipeline.load_and_preprocess(str(file_path))
        slices = pipeline.generate_report_slices(result, volume, num_slices=5)
        if slices:
            from PIL import Image

            for i, s in enumerate(slices):
                img = Image.fromarray(s)
                img.save(str(results_subdir / f"slice_{i}.png"))
            heatmap_path = str(results_subdir / "slice_0.png")

    except Exception as e:
        logger.warning(f"Failed to save artifacts: {e}")

    # Store in database
    async with _get_session(request) as session:
        try:
            # Get or create patient
            stmt = select(Patient).where(Patient.patient_id == patient_id)
            db_result = await session.execute(stmt)
            patient = db_result.scalar_one_or_none()

            if patient is None:
                patient = Patient(
                    patient_id=patient_id,
                    patient_name=patient_name,
                )
                session.add(patient)
                await session.flush()

            # Create scan record
            ct_scan = CTScan(
                id=scan_id,
                patient_id=patient.id,
                file_path=str(file_path),
                file_type=file_type,
                study_date=result.get("metadata", {}).get("study_date"),
                scan_metadata=result.get("metadata"),
            )
            session.add(ct_scan)

            # Create analysis result
            change = result.get("change")
            analysis = AnalysisResult(
                scan_id=scan_id,
                severity=result["severity"],
                severity_label=result["severity_label"],
                confidence=result["confidence"],
                damage_percent=result["damage_percent"],
                probabilities={"classes": result.get("probabilities", [])},
                change_class=change["change_class"] if change else None,
                change_label=change["change_label"] if change else None,
                change_score=change["change_score"] if change else None,
                baseline_scan_id=baseline_scan_id,
                segmentation_path=seg_path,
                heatmap_path=heatmap_path,
                difference_map_path=diff_path,
                processing_time=result.get("total_time"),
                stage_times=result.get("stages"),
            )
            session.add(analysis)
            await session.commit()
            logger.info(f"Stored analysis for scan {scan_id}")

        except Exception as e:
            await session.rollback()
            logger.error(f"Database error: {e}", exc_info=True)
            raise HTTPException(500, f"Failed to store results: {e}")

    return UploadResponse(
        scan_id=scan_id,
        patient_id=patient_id,
        severity=result["severity_label"],
        damage_percent=result["damage_percent"],
        change=change["change_label"] if change else None,
        processing_time=result.get("total_time", 0.0),
    )


@router.get("/patients", response_model=list[PatientResponse])
async def list_patients(request: Request, skip: int = 0, limit: int = 50):
    """List all patients."""
    async with _get_session(request) as session:
        stmt = (
            select(Patient)
            .order_by(Patient.updated_at.desc())
            .offset(skip)
            .limit(limit)
        )
        result = await session.execute(stmt)
        patients = result.scalars().all()

        responses = []
        for p in patients:
            scan_count_stmt = select(CTScan).where(CTScan.patient_id == p.id)
            count_result = await session.execute(scan_count_stmt)
            scan_count = len(count_result.scalars().all())
            responses.append(
                PatientResponse(
                    id=p.id,
                    patient_id=p.patient_id,
                    patient_name=p.patient_name,
                    created_at=p.created_at,
                    scan_count=scan_count,
                )
            )
        return responses


@router.get("/patients/{patient_id}/scans", response_model=PatientDetailResponse)
async def get_patient_scans(request: Request, patient_id: str):
    """Get patient details with all scan history."""
    async with _get_session(request) as session:
        stmt = (
            select(Patient)
            .where(Patient.patient_id == patient_id)
            .options(selectinload(Patient.scans).selectinload(CTScan.result))
        )
        result = await session.execute(stmt)
        patient = result.scalar_one_or_none()

        if patient is None:
            raise HTTPException(404, f"Patient not found: {patient_id}")

        return PatientDetailResponse(
            id=patient.id,
            patient_id=patient.patient_id,
            patient_name=patient.patient_name,
            created_at=patient.created_at,
            scan_count=len(patient.scans),
            scans=[
                ScanResponse(
                    id=s.id,
                    patient_id=s.patient_id,
                    file_type=s.file_type,
                    study_date=s.study_date,
                    uploaded_at=s.uploaded_at,
                    result=s.result,
                )
                for s in sorted(patient.scans, key=lambda x: x.uploaded_at, reverse=True)
            ],
        )


@router.get("/patients/{patient_id}/damage-history", response_model=DamageHistoryResponse)
async def get_damage_history(request: Request, patient_id: str):
    """Get longitudinal damage percentage history for a patient."""
    async with _get_session(request) as session:
        stmt = (
            select(Patient)
            .where(Patient.patient_id == patient_id)
            .options(selectinload(Patient.scans).selectinload(CTScan.result))
        )
        result = await session.execute(stmt)
        patient = result.scalar_one_or_none()

        if patient is None:
            raise HTTPException(404, f"Patient not found: {patient_id}")

        history = []
        for scan in sorted(patient.scans, key=lambda x: x.uploaded_at):
            if scan.result:
                history.append(
                    DamageHistoryPoint(
                        scan_id=scan.id,
                        study_date=scan.study_date,
                        damage_percent=scan.result.damage_percent,
                        severity_label=scan.result.severity_label,
                        uploaded_at=scan.uploaded_at,
                    )
                )

        return DamageHistoryResponse(patient_id=patient_id, history=history)
