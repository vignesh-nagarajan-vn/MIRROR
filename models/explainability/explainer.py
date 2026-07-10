"""High-level explainer used by the pipeline.

Wraps the chosen CAM method (Grad-CAM or Score-CAM), resolves the right target
layer for the backbone, and produces a ready-to-display overlay plus a small
amount of structured metadata (the centroid and bounding box of the activated
region) that the report generator can describe in words.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..common.preprocessing import preprocess, denormalize
from ..classification.model import resolve_target_layer, DEFAULT_TARGET_LAYERS
from .gradcam import GradCAM, vit_reshape_transform
from .scorecam import ScoreCAM
from .overlay import overlay_heatmap, to_png_bytes


@dataclass
class Explanation:
    label: str
    class_idx: int
    heatmap: np.ndarray          # (H, W) float in [0, 1]
    overlay_png: bytes           # rendered overlay, PNG-encoded
    centroid: tuple[float, float]  # (x, y) normalised 0-1 of peak activation
    bbox: tuple[float, float, float, float]  # x0, y0, x1, y1 normalised


def _region_stats(cam: np.ndarray, threshold: float = 0.5):
    """Return the centroid and bounding box of the activated region."""
    h, w = cam.shape
    mask = cam >= threshold
    if not mask.any():
        return (0.5, 0.5), (0.0, 0.0, 1.0, 1.0)
    ys, xs = np.where(mask)
    cx = float(xs.mean() / w)
    cy = float(ys.mean() / h)
    bbox = (
        float(xs.min() / w),
        float(ys.min() / h),
        float(xs.max() / w),
        float(ys.max() / h),
    )
    return (cx, cy), bbox


class Explainer:
    """Produce visual explanations for a classifier's prediction."""

    def __init__(
        self,
        model,
        backbone: str = "densenet121",
        method: str = "gradcam",
        target_layer: str | None = None,
        image_size: int = 224,
        overlay_alpha: float = 0.45,
        colormap: str = "jet",
    ) -> None:
        self.model = model
        self.backbone = backbone
        self.method = method
        self.image_size = image_size
        self.overlay_alpha = overlay_alpha
        self.colormap = colormap
        self.layer = resolve_target_layer(model, backbone, target_layer)
        self.reshape = (
            vit_reshape_transform if backbone.lower().startswith("vit") else None
        )

    def _build_cam(self):
        if self.method == "scorecam":
            return ScoreCAM(self.model, self.layer)
        return GradCAM(self.model, self.layer, reshape_transform=self.reshape)

    def explain(self, source, label: str, class_idx: int) -> Explanation:
        tensor, _ = preprocess(source, self.image_size)
        tensor = tensor.to(next(self.model.parameters()).device)

        with self._build_cam() as cam_fn:
            cam = cam_fn(tensor, class_idx=class_idx)

        base = denormalize(tensor)
        overlay = overlay_heatmap(base, cam, self.overlay_alpha, self.colormap)
        centroid, bbox = _region_stats(cam)

        return Explanation(
            label=label,
            class_idx=class_idx,
            heatmap=cam,
            overlay_png=to_png_bytes(overlay),
            centroid=centroid,
            bbox=bbox,
        )


def _describe_frontal(x: float, y: float) -> str:
    """Chest-radiograph vocabulary: a 3x3 grid of lung zones."""
    vertical = "upper" if y < 0.33 else "mid" if y < 0.66 else "lower"
    # Radiology convention: patient's right is on the viewer's left.
    horizontal = "right" if x < 0.33 else "central" if x < 0.66 else "left"
    if horizontal == "central":
        return f"the {vertical} central zone"
    return f"the {vertical} {horizontal} zone"


def _describe_axial(x: float, y: float) -> str:
    """Cross-sectional (brain MRI / head CT) vocabulary: lobar regions.

    On an axial slice displayed in radiological convention the patient's right is
    on the viewer's left (same flip as a frontal film), and the top of the image
    is anterior. We map the vertical axis to anterior->posterior lobes and the
    horizontal axis to laterality.
    """
    region = "frontal" if y < 0.33 else "parietal" if y < 0.66 else "occipital"
    side = "right" if x < 0.33 else "midline" if x < 0.66 else "left"
    if side == "midline":
        return f"the {region} midline region"
    return f"the {side} {region} region"


def describe_location(centroid: tuple[float, float], plane: str = "frontal") -> str:
    """Convert a normalised centroid into anatomical-ish plain English.

    ``plane`` selects the vocabulary: ``"frontal"`` (a chest film -> lung zones)
    or ``"axial"`` (a cross-sectional brain MRI / head CT -> lobar regions). A
    rough 3x3 mapping in both cases: enough to ground the language model and give
    a clinician a quick orientation. Defaults to ``"frontal"`` for backward
    compatibility with the original chest-only pipeline.
    """
    x, y = centroid
    if plane == "axial":
        return _describe_axial(x, y)
    return _describe_frontal(x, y)
