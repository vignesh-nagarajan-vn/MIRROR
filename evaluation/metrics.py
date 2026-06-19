"""Evaluation metrics for MIRROR.

Two families of metrics, reflecting the project's research question of whether
explanation/reasoning layers help *without* sacrificing predictive quality:

1. Predictive performance — per-label and macro AUROC, F1 at a chosen threshold.
2. Explanation quality — pointing game and localization IoU when ground-truth
   bounding boxes are available (e.g. the ~1,000 boxed ChestX-ray14 images).
"""

from __future__ import annotations

import numpy as np

try:
    from sklearn.metrics import roc_auc_score, f1_score
except ImportError:  # pragma: no cover
    roc_auc_score = None
    f1_score = None


def macro_auroc(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    """Per-label AUROC plus the macro average.

    ``y_true`` and ``y_score`` are (N, C). Labels with a single class present in
    the batch are skipped (AUROC undefined).
    """
    if roc_auc_score is None:
        raise RuntimeError("scikit-learn is required for AUROC.")
    per_label: dict[str, float] = {}
    valid = []
    for c in range(y_true.shape[1]):
        if len(np.unique(y_true[:, c])) > 1:
            auc = float(roc_auc_score(y_true[:, c], y_score[:, c]))
            per_label[str(c)] = auc
            valid.append(auc)
    per_label["macro"] = float(np.mean(valid)) if valid else 0.0
    return per_label


def f1_at_threshold(
    y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5
) -> float:
    """Macro F1 after binarizing scores at ``threshold``."""
    if f1_score is None:
        raise RuntimeError("scikit-learn is required for F1.")
    y_pred = (y_score >= threshold).astype(int)
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def pointing_game(cam: np.ndarray, gt_box: tuple[int, int, int, int]) -> bool:
    """Pointing game hit: does the CAM's peak fall inside the GT box?

    ``gt_box`` is (x0, y0, x1, y1) in pixel coordinates of the CAM grid.
    """
    y_peak, x_peak = np.unravel_index(int(cam.argmax()), cam.shape)
    x0, y0, x1, y1 = gt_box
    return (x0 <= x_peak <= x1) and (y0 <= y_peak <= y1)


def localization_iou(
    cam: np.ndarray, gt_box: tuple[int, int, int, int], threshold: float = 0.5
) -> float:
    """IoU between the thresholded CAM region and a ground-truth box."""
    mask = cam >= threshold
    gt = np.zeros_like(cam, dtype=bool)
    x0, y0, x1, y1 = gt_box
    gt[y0:y1, x0:x1] = True
    inter = np.logical_and(mask, gt).sum()
    union = np.logical_or(mask, gt).sum()
    return float(inter / union) if union > 0 else 0.0
