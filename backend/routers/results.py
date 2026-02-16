"""
Results & visualization API routes.

GET /api/results/{scan_id}           — Get analysis result
GET /api/results/{scan_id}/heatmap   — Get difference heatmap image
GET /api/results/{scan_id}/segmentation — Get segmentation overlay image
GET /api/results/{scan_id}/slices    — Get report slice images
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import APIRouter, HTTPException, Request, Query
from fastapi.responses import FileResponse, JSONResponse

from backend.config import settings
from backend.models import CTScan, AnalysisResult
from backend.schemas import AnalysisResponse, ChangeResponse

logger = logging.getLogger(__name__)
router = APIRouter()


def _get_session(request: Request):
    """Get async session from app state."""
    return request.app.state.session_factory()


@router.get("/results/{scan_id}", response_model=AnalysisResponse)
async def get_result(request: Request, scan_id: str):
    """
    Get the analysis result for a specific CT scan.

    Returns severity classification, damage percentage,
    and longitudinal change if available.
    """
    async with _get_session(request) as session:
        result = await session.get(AnalysisResult, None)

        # Query by scan_id
        from sqlalchemy import select
        stmt = select(AnalysisResult).where(AnalysisResult.scan_id == scan_id)
        db_result = await session.execute(stmt)
        analysis = db_result.scalar_one_or_none()

        if analysis is None:
            raise HTTPException(404, f"No analysis found for scan: {scan_id}")

        change = None
        if analysis.change_label is not None:
            change = ChangeResponse(
                change_class=analysis.change_class,
                change_label=analysis.change_label,
                change_score=analysis.change_score,
            )

        return AnalysisResponse(
            id=analysis.id,
            scan_id=analysis.scan_id,
            severity=analysis.severity,
            severity_label=analysis.severity_label,
            confidence=analysis.confidence,
            damage_percent=analysis.damage_percent,
            probabilities=analysis.probabilities,
            change=change,
            processing_time=analysis.processing_time,
            stage_times=analysis.stage_times,
            created_at=analysis.created_at,
        )


@router.get("/results/{scan_id}/heatmap")
async def get_heatmap(
    request: Request,
    scan_id: str,
    slice_idx: int = Query(0, ge=0, le=4, description="Slice index (0-4)"),
):
    """
    Get a heatmap overlay image for the analysis result.

    Returns a PNG image of the specified axial slice with
    the difference map or segmentation overlay.
    """
    results_dir = Path(settings.results_dir) / scan_id
    slice_file = results_dir / f"slice_{slice_idx}.png"

    if not slice_file.exists():
        raise HTTPException(
            404,
            f"Heatmap not found for scan {scan_id}, slice {slice_idx}. "
            f"Available slices: {_list_available_slices(results_dir)}",
        )

    return FileResponse(
        str(slice_file),
        media_type="image/png",
        filename=f"heatmap_{scan_id}_slice{slice_idx}.png",
    )


@router.get("/results/{scan_id}/segmentation")
async def get_segmentation(request: Request, scan_id: str):
    """
    Get the segmentation mask as a downloadable numpy file.

    The mask is a 3D binary array (D, H, W) with 1 = lung, 0 = background.
    """
    seg_path = Path(settings.results_dir) / scan_id / "segmentation.npy"

    if not seg_path.exists():
        raise HTTPException(404, f"Segmentation not found for scan: {scan_id}")

    return FileResponse(
        str(seg_path),
        media_type="application/octet-stream",
        filename=f"segmentation_{scan_id}.npy",
    )


@router.get("/results/{scan_id}/difference-map")
async def get_difference_map(request: Request, scan_id: str):
    """
    Get the registration difference map as a downloadable numpy file.

    Only available if the scan was compared against a baseline.
    """
    diff_path = Path(settings.results_dir) / scan_id / "difference_map.npy"

    if not diff_path.exists():
        raise HTTPException(
            404,
            f"Difference map not found for scan: {scan_id}. "
            "This scan may not have been compared against a baseline.",
        )

    return FileResponse(
        str(diff_path),
        media_type="application/octet-stream",
        filename=f"difference_map_{scan_id}.npy",
    )


@router.get("/results/{scan_id}/slices")
async def list_slices(request: Request, scan_id: str):
    """
    List available report slice images for a scan.

    Returns URLs for each available heatmap slice.
    """
    results_dir = Path(settings.results_dir) / scan_id
    available = _list_available_slices(results_dir)

    if not available:
        raise HTTPException(404, f"No slices found for scan: {scan_id}")

    base_url = f"/api/results/{scan_id}/heatmap"
    return {
        "scan_id": scan_id,
        "slices": [
            {"index": i, "url": f"{base_url}?slice_idx={i}"}
            for i in available
        ],
    }


@router.get("/results/{scan_id}/summary")
async def get_result_summary(request: Request, scan_id: str):
    """
    Get a concise JSON summary suitable for the frontend dashboard.
    """
    async with _get_session(request) as session:
        from sqlalchemy import select
        stmt = select(AnalysisResult).where(AnalysisResult.scan_id == scan_id)
        db_result = await session.execute(stmt)
        analysis = db_result.scalar_one_or_none()

        if analysis is None:
            raise HTTPException(404, f"No analysis found for scan: {scan_id}")

        results_dir = Path(settings.results_dir) / scan_id
        available_slices = _list_available_slices(results_dir)

        return {
            "severity": analysis.severity_label,
            "lung_damage_percent": round(analysis.damage_percent, 1),
            "change": analysis.change_label or "N/A",
            "confidence": round(analysis.confidence * 100, 1),
            "processing_time_sec": round(analysis.processing_time or 0, 2),
            "heatmap_slices": len(available_slices),
            "has_segmentation": (results_dir / "segmentation.npy").exists(),
            "has_difference_map": (results_dir / "difference_map.npy").exists(),
        }


def _list_available_slices(results_dir: Path) -> list[int]:
    """List available slice indices in a results directory."""
    if not results_dir.exists():
        return []
    return sorted(
        int(f.stem.split("_")[1])
        for f in results_dir.glob("slice_*.png")
    )
