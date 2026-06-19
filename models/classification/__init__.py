"""Image classification layer for MIRROR."""

from .model import build_model, resolve_target_layer, SUPPORTED_BACKBONES
from .infer import Classifier, Prediction

__all__ = [
    "build_model",
    "resolve_target_layer",
    "SUPPORTED_BACKBONES",
    "Classifier",
    "Prediction",
]
