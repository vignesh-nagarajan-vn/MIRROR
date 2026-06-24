"""Configuration objects for MIRROR.

Configuration is intentionally simple: a dataclass with sane defaults that can be
overridden from a YAML file (`configs/default.yaml`) or environment variables.
This avoids a heavyweight config framework while still keeping every tunable in
one place.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover - yaml is a declared dependency
    yaml = None


@dataclass
class ModelConfig:
    """Backbone + classifier head settings."""

    backbone: str = "densenet121"  # one of: densenet121, efficientnet_b0, vit_b_16
    num_classes: int = 14
    pretrained: bool = True
    dropout: float = 0.2
    checkpoint_path: str | None = None


@dataclass
class ExplainConfig:
    """Visual explanation settings."""

    method: str = "gradcam"  # gradcam | scorecam
    target_layer: str | None = None  # auto-resolved per backbone if None
    overlay_alpha: float = 0.45
    colormap: str = "jet"


@dataclass
class ReportConfig:
    """LLM report generation settings."""

    provider: str = "anthropic"  # anthropic | template (offline fallback)
    model: str = "claude-haiku-4-5"
    max_tokens: int = 1024
    temperature: float = 0.2
    confidence_threshold: float = 0.5  # findings below this are reported as "no acute"


@dataclass
class DataConfig:
    """Dataset locations and loader settings."""

    dataset: str = "chestxray14"
    data_root: str = "datasets/raw/chestxray14"
    image_size: int = 224
    batch_size: int = 32
    num_workers: int = 4


@dataclass
class TrainConfig:
    """Optimisation settings."""

    epochs: int = 15
    lr: float = 1e-4
    weight_decay: float = 1e-5
    device: str = "cuda"  # falls back to cpu automatically at runtime
    output_dir: str = "models/checkpoints"
    seed: int = 42


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    explain: ExplainConfig = field(default_factory=ExplainConfig)
    report: ReportConfig = field(default_factory=ReportConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        if yaml is None:
            raise RuntimeError("PyYAML is required to load config files.")
        with open(path, "r", encoding="utf-8") as fh:
            raw: dict[str, Any] = yaml.safe_load(fh) or {}
        return cls(
            model=ModelConfig(**raw.get("model", {})),
            explain=ExplainConfig(**raw.get("explain", {})),
            report=ReportConfig(**raw.get("report", {})),
            data=DataConfig(**raw.get("data", {})),
            train=TrainConfig(**raw.get("train", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def get_anthropic_api_key() -> str | None:
    """Read the Anthropic API key from the environment.

    Never hard-code keys. The backend and report generator both call this.
    """
    return os.environ.get("ANTHROPIC_API_KEY")
