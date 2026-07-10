"""API routes for MIRROR.

Endpoints
---------
GET  /api/health        liveness + whether the model is loaded
GET  /api/labels        the label set for a modality (default chest X-ray)
GET  /api/modalities    every supported modality + its label set
POST /api/analyze       upload an image, get predictions + overlays + report
"""

from __future__ import annotations

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.core.config import get_settings
from app.schemas.responses import AnalysisResponse, HealthResponse
from app.services.pipeline_service import pipeline_service
from models.common.modalities import list_modalities, resolve_modality

router = APIRouter(prefix="/api")

ALLOWED_CONTENT_TYPES = {"image/png", "image/jpeg", "image/jpg", "image/bmp", "image/webp"}
# DICOM is the native radiology format. Browsers and PACS exporters are
# inconsistent here: some send "application/dicom", many send a generic
# "application/octet-stream", so we also accept by filename / magic bytes below.
DICOM_CONTENT_TYPES = {"application/dicom", "application/octet-stream", ""}
DICOM_EXTENSIONS = (".dcm", ".dicom", ".dco")


def _looks_like_dicom(image: UploadFile) -> bool:
    name = (image.filename or "").lower()
    if name.endswith(DICOM_EXTENSIONS):
        return True
    return image.content_type in {"application/dicom"}


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    settings = get_settings()
    return HealthResponse(
        status="ok",
        version=settings.version,
        model_loaded=pipeline_service.is_loaded,
        backbone=pipeline_service.backbone,
    )


@router.get("/labels")
def labels(modality: str = "chest X-ray") -> dict:
    """Label set for a modality (query ``?modality=brain MRI`` to switch)."""
    spec = resolve_modality(modality)
    return {
        "modality": spec.display_name,
        "modality_key": spec.key,
        "labels": list(spec.labels),
        "count": spec.num_labels,
    }


@router.get("/modalities")
def modalities() -> dict:
    """Every supported modality and its label set (for the UI selector)."""
    return {"modalities": list_modalities()}


@router.post("/analyze", response_model=AnalysisResponse)
async def analyze(
    image: UploadFile = File(..., description="Radiograph (PNG/JPEG/DICOM)."),
    modality: str = Form("chest X-ray"),
    indication: str | None = Form(None),
) -> AnalysisResponse:
    is_dicom_upload = _looks_like_dicom(image)
    if not is_dicom_upload and image.content_type not in ALLOWED_CONTENT_TYPES:
        # Allow generic octet-streams only when they turn out to be DICOM (checked
        # against the magic bytes after reading); reject anything else up front.
        if image.content_type not in DICOM_CONTENT_TYPES:
            raise HTTPException(
                status_code=415,
                detail=f"Unsupported image type '{image.content_type}'. "
                "Accepted: PNG, JPEG, BMP, WEBP, or DICOM (.dcm).",
            )

    settings = get_settings()
    raw = await image.read()
    if len(raw) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail=f"Image exceeds {settings.max_upload_mb} MB limit.",
        )

    try:
        result = pipeline_service.analyze(raw, modality=modality, indication=indication)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Analysis failed: {exc}") from exc

    return AnalysisResponse(**result.to_dict())
