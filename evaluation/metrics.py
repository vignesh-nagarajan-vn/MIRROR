"""Evaluation metrics for MIRROR.

Two families of metrics, reflecting the project's research question of whether
explanation/reasoning layers help *without* sacrificing predictive quality:

1. Predictive performance — per-label and macro AUROC, F1 at a chosen threshold,
   each with a bootstrap confidence interval over the test set.
2. Explanation quality — pointing game and localization IoU when ground-truth
   bounding boxes are available (e.g. the ~1,000 boxed ChestX-ray14 images).

Point estimates alone are not enough to claim one configuration beats another:
``bootstrap_cis`` resamples the test set to attach 95% CIs to the headline
numbers, and ``evaluation/aggregate_seeds.py`` summarises across training seeds.
"""

from __future__ import annotations

from collections import defaultdict

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


def _percentile_ci(samples: list[float], ci: float) -> dict[str, float]:
    """Mean, std, and a two-sided percentile interval over bootstrap samples."""
    if not samples:
        return {"mean": 0.0, "std": 0.0, "lo": 0.0, "hi": 0.0}
    arr = np.asarray(samples, dtype=float)
    lo_pct = (1.0 - ci) / 2.0 * 100.0
    hi_pct = (1.0 + ci) / 2.0 * 100.0
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "lo": float(np.percentile(arr, lo_pct)),
        "hi": float(np.percentile(arr, hi_pct)),
    }


def bootstrap_cis(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float = 0.5,
    n_boot: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> dict:
    """Bootstrap confidence intervals for macro AUROC, macro F1, per-label AUROC.

    Resamples the N examples with replacement ``n_boot`` times (one resample
    drives all metrics, so the CIs are mutually consistent) and reports a
    two-sided ``ci`` percentile interval. Per-label AUROC is only collected from
    resamples in which that label has both classes present, mirroring
    ``macro_auroc``. Deterministic given ``seed``.
    """
    rng = np.random.default_rng(seed)
    n = y_true.shape[0]

    macro_auroc_samples: list[float] = []
    macro_f1_samples: list[float] = []
    per_label_samples: dict[str, list[float]] = defaultdict(list)

    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt, ys = y_true[idx], y_score[idx]
        auroc = macro_auroc(yt, ys)
        macro_auroc_samples.append(auroc["macro"])
        for key, value in auroc.items():
            if key != "macro":
                per_label_samples[key].append(value)
        macro_f1_samples.append(f1_at_threshold(yt, ys, threshold))

    return {
        "n_boot": n_boot,
        "ci_level": ci,
        "seed": seed,
        "macro_auroc": _percentile_ci(macro_auroc_samples, ci),
        "macro_f1": _percentile_ci(macro_f1_samples, ci),
        "per_label_auroc": {
            key: _percentile_ci(vals, ci) for key, vals in per_label_samples.items()
        },
    }
