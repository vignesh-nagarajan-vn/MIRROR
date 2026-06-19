"""MIRROR FastAPI application.

Run locally:
    uvicorn app.main:app --reload --port 8000

Interactive docs are served at /docs (Swagger) and /redoc.
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.core.config import get_settings


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title=settings.app_name,
        version=settings.version,
        description=(
            "Multimodal Intelligent Radiology Reasoning and Observation Reporter. "
            "Upload a radiograph to receive disease predictions, Grad-CAM evidence "
            "overlays, and a draft clinician-style report. Research use only — not "
            "a medical device."
        ),
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_origins),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router)

    @app.get("/")
    def root() -> dict:
        return {
            "name": settings.app_name,
            "version": settings.version,
            "docs": "/docs",
            "disclaimer": "Research prototype. Not for clinical use.",
        }

    return app


app = create_app()
