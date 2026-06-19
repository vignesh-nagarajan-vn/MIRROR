"""Backend settings.

Reads configuration from environment variables with sensible defaults, so the
same image runs in dev and prod without code changes. Secrets (the Anthropic
key) are never defaulted to a real value.
"""

from __future__ import annotations

import os
from functools import lru_cache
from dataclasses import dataclass


@dataclass
class Settings:
    app_name: str = "MIRROR API"
    version: str = "0.1.0"
    config_path: str = os.environ.get("MIRROR_CONFIG", "configs/default.yaml")
    # Comma-separated list of allowed CORS origins for the Next.js frontend.
    cors_origins: tuple[str, ...] = tuple(
        o.strip()
        for o in os.environ.get(
            "MIRROR_CORS_ORIGINS", "http://localhost:3000"
        ).split(",")
        if o.strip()
    )
    max_upload_mb: int = int(os.environ.get("MIRROR_MAX_UPLOAD_MB", "20"))


@lru_cache
def get_settings() -> Settings:
    return Settings()
