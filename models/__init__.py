"""MIRROR model package: classification, explainability, and report generation.

The headline export is :class:`MirrorPipeline`, which runs the full
Image -> Prediction -> Evidence -> Report flow.
"""

from .pipeline import MirrorPipeline, AnalysisResult, FindingResult
from .common.config import Config

__all__ = ["MirrorPipeline", "AnalysisResult", "FindingResult", "Config"]
