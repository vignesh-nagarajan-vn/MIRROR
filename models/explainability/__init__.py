"""Visual explainability layer for MIRROR (Grad-CAM, Score-CAM)."""

from .gradcam import GradCAM, vit_reshape_transform
from .scorecam import ScoreCAM
from .explainer import Explainer, Explanation, describe_location
from .overlay import overlay_heatmap, to_png_bytes

__all__ = [
    "GradCAM",
    "ScoreCAM",
    "vit_reshape_transform",
    "Explainer",
    "Explanation",
    "describe_location",
    "overlay_heatmap",
    "to_png_bytes",
]
