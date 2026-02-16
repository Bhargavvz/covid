"""
Pydantic schemas for API request/response validation.
"""

from datetime import datetime
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field


# ── Request Schemas ────────────────────────────────────────────

class UploadRequest(BaseModel):
    """Metadata for CT scan upload."""
    patient_id: str = Field(..., description="External patient identifier")
    patient_name: Optional[str] = Field(None, description="Patient display name")
    baseline_scan_id: Optional[str] = Field(
        None, description="Scan ID of baseline for longitudinal comparison"
    )


# ── Response Schemas ───────────────────────────────────────────

class SeverityResponse(BaseModel):
    """Severity classification result."""
    severity: int = Field(..., ge=0, le=3, description="Severity class (0-3)")
    severity_label: str = Field(..., description="Severity label")
    confidence: float = Field(..., ge=0, le=1, description="Classification confidence")
    damage_percent: float = Field(..., ge=0, le=100, description="Estimated % lung damage")
    probabilities: Optional[Dict[str, float]] = None


class ChangeResponse(BaseModel):
    """Longitudinal change result."""
    change_class: int = Field(..., description="Change class (0=improved, 1=stable, 2=worsened)")
    change_label: str = Field(..., description="Change label")
    change_score: float = Field(..., description="Change magnitude score")


class AnalysisResponse(BaseModel):
    """Full analysis result for a CT scan."""
    id: str
    scan_id: str
    severity: int
    severity_label: str
    confidence: float
    damage_percent: float
    probabilities: Optional[Dict[str, float]] = None
    change: Optional[ChangeResponse] = None
    processing_time: Optional[float] = None
    stage_times: Optional[Dict[str, float]] = None
    created_at: datetime

    class Config:
        from_attributes = True


class ScanResponse(BaseModel):
    """CT scan record."""
    id: str
    patient_id: str
    file_type: str
    study_date: Optional[str] = None
    uploaded_at: datetime
    result: Optional[AnalysisResponse] = None

    class Config:
        from_attributes = True


class PatientResponse(BaseModel):
    """Patient record with scan history."""
    id: str
    patient_id: str
    patient_name: Optional[str] = None
    created_at: datetime
    scan_count: int = 0

    class Config:
        from_attributes = True


class PatientDetailResponse(PatientResponse):
    """Patient with full scan history."""
    scans: List[ScanResponse] = []


class UploadResponse(BaseModel):
    """Response after uploading and analyzing a CT scan."""
    scan_id: str
    patient_id: str
    severity: str
    damage_percent: float
    change: Optional[str] = None
    processing_time: float
    message: str = "Analysis complete"


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = "healthy"
    gpu_available: bool = False
    models_loaded: Dict[str, bool] = {}
    version: str = "1.0.0"


class DamageHistoryPoint(BaseModel):
    """Single point in damage history timeline."""
    scan_id: str
    study_date: Optional[str]
    damage_percent: float
    severity_label: str
    uploaded_at: datetime


class DamageHistoryResponse(BaseModel):
    """Longitudinal damage history for a patient."""
    patient_id: str
    history: List[DamageHistoryPoint] = []
