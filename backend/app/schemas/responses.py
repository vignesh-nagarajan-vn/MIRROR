"""Pydantic schemas for the MIRROR API responses."""

from __future__ import annotations

from pydantic import BaseModel, Field


class FindingSchema(BaseModel):
    label: str = Field(..., description="Disease/finding name (ChestX-ray14 taxonomy).")
    probability: float = Field(..., ge=0.0, le=1.0)
    present: bool = Field(..., description="True if probability >= threshold.")
    location: str = Field(..., description="Plain-English zone from the saliency map.")
    overlay_png_b64: str | None = Field(
        None, description="Base64-encoded PNG of the Grad-CAM overlay, if computed."
    )


class AnalysisResponse(BaseModel):
    modality: str
    backbone: str
    explain_method: str
    report: str
    report_backend: str
    findings: list[FindingSchema]
    meta: dict


class HealthResponse(BaseModel):
    status: str
    version: str
    model_loaded: bool
    backbone: str | None = None
