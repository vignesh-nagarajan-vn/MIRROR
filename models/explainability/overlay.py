"""Heatmap overlay rendering for MIRROR explanations.

Turns a normalised [0, 1] CAM into a coloured overlay on top of the original
image and returns it as a PNG byte string, suitable for sending to the frontend
or saving to disk.
"""

from __future__ import annotations

from io import BytesIO

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")  # headless rendering
    import matplotlib.cm as cm
    from PIL import Image
except ImportError:  # pragma: no cover
    cm = None
    Image = None


def apply_colormap(cam: np.ndarray, colormap: str = "jet") -> np.ndarray:
    """Map a (H, W) heatmap in [0, 1] to an (H, W, 3) uint8 RGB array."""
    if cm is None:
        raise RuntimeError("matplotlib is required for colormaps.")
    mapper = cm.get_cmap(colormap)
    colored = mapper(cam)[:, :, :3]  # drop alpha
    return (colored * 255).astype(np.uint8)


def overlay_heatmap(
    base_image: np.ndarray,
    cam: np.ndarray,
    alpha: float = 0.45,
    colormap: str = "jet",
) -> np.ndarray:
    """Blend a heatmap onto a base RGB image (both HxWx3 uint8-compatible)."""
    if Image is None:
        raise RuntimeError("Pillow is required for overlays.")
    heat = apply_colormap(cam, colormap).astype(np.float32)
    base = base_image.astype(np.float32)
    blended = (1 - alpha) * base + alpha * heat
    return np.clip(blended, 0, 255).astype(np.uint8)


def to_png_bytes(image: np.ndarray) -> bytes:
    """Encode an HxWx3 uint8 array as PNG bytes."""
    if Image is None:
        raise RuntimeError("Pillow is required.")
    buffer = BytesIO()
    Image.fromarray(image).save(buffer, format="PNG")
    return buffer.getvalue()
