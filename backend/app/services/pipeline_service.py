"""Pipeline service for the backend.

Loads the (potentially heavy) MirrorPipeline lazily and exactly once, then reuses
it across requests. Loading is deferred until the first analysis call so the API
can start and report healthy quickly even on machines where model init is slow.
"""

from __future__ import annotations

from threading import Lock

from models.pipeline import MirrorPipeline, AnalysisResult
from models.common.config import Config

from app.core.config import get_settings


class PipelineService:
    def __init__(self) -> None:
        self._pipeline: MirrorPipeline | None = None
        self._lock = Lock()

    @property
    def is_loaded(self) -> bool:
        return self._pipeline is not None

    @property
    def backbone(self) -> str | None:
        return self._pipeline.config.model.backbone if self._pipeline else None

    def _ensure_loaded(self) -> MirrorPipeline:
        if self._pipeline is None:
            with self._lock:
                if self._pipeline is None:  # double-checked locking
                    settings = get_settings()
                    try:
                        config = Config.from_yaml(settings.config_path)
                    except Exception:  # noqa: BLE001 - fall back to defaults
                        config = Config()
                    self._pipeline = MirrorPipeline(config)
        return self._pipeline

    def analyze(
        self,
        image_bytes: bytes,
        modality: str = "chest X-ray",
        indication: str | None = None,
    ) -> AnalysisResult:
        pipeline = self._ensure_loaded()
        return pipeline.analyze(image_bytes, modality=modality, indication=indication)


# Module-level singleton.
pipeline_service = PipelineService()
