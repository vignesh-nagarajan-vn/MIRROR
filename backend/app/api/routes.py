"""API routes for MIRROR.

Endpoints
---------
GET  /api/health        liveness + whether the model is loaded
GET  /api/labels        the ChestX-ray14 label set
POST /api/analyze       upload an image, get predictions + overlays + report
"""

from __future__ import annotations

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.core.config import get_settings
from app.schemas.responses import AnalysisResponse, HealthResponse
from app.services.pipeline_service import pipeline_service
from models.common.constants import CHESTXRAY14_LABELS

router = APIRouter(prefix="/api")

ALLOWED_CONTENT_TYPES = {"image/png", "image/jpeg", "image/jpg", "image/bmp", "image/webp"}


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
def labels() -> dict:
    return {"labels": CHESTXRAY14_LABELS, "count": len(CHESTXRAY14_LABELS)}


@router.post("/analyze", response_model=AnalysisResponse)
async def analyze(
    image: UploadFile = File(..., description="Radiograph (PNG/JPEG)."),
    modality: str = Form("chest X-ray"),
    indication: str | None = Form(None),
) -> AnalysisResponse:
    if image.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported image type '{image.content_type}'.",
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
