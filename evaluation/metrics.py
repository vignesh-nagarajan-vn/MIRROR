"""Evaluation metrics for MIRROR.

Three families of metrics, reflecting the project's research question of whether
explanation/reasoning layers help *without* sacrificing predictive quality, and
the standard a clinical reader would expect of a diagnostic model:

1. Discrimination — per-label and macro AUROC, macro AUPRC, F1 at a chosen
   threshold, each with a bootstrap confidence interval over the test set.
2. Operating-point performance — sensitivity, specificity, PPV, and NPV at the
   decision threshold (the quantities a clinician actually reads off), per label
   and macro-averaged, with positive support counts.
3. Calibration — Brier score and Expected Calibration Error, because a
   probability a clinician is asked to trust must mean what it says.
4. Explanation quality — pointing game and localization IoU when ground-truth
   bounding boxes are available (e.g. the ~1,000 boxed ChestX-ray14 images).

Point estimates alone are not enough to claim one configuration beats another:
``bootstrap_cis`` resamples the test set to attach 95% CIs to the headline
numbers, and ``evaluation/aggregate_seeds.py`` summarises across training seeds.

The confusion-matrix and calibration metrics are pure NumPy (no scikit-learn), so
they stay importable — and unit-testable — without the ML stack.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

try:
    from sklearn.metrics import roc_auc_score, f1_score, average_precision_score
except ImportError:  # pragma: no cover
    roc_auc_score = None
    f1_score = None
    average_precision_score = None


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


def macro_auprc(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    """Per-label AUPRC (average precision) plus the macro average.

    AUPRC is the more informative discrimination metric under the heavy class
    imbalance typical of medical findings, where AUROC can look optimistic.
    """
    if average_precision_score is None:
        raise RuntimeError("scikit-learn is required for AUPRC.")
    per_label: dict[str, float] = {}
    valid = []
    for c in range(y_true.shape[1]):
        if len(np.unique(y_true[:, c])) > 1:
            ap = float(average_precision_score(y_true[:, c], y_score[:, c]))
            per_label[str(c)] = ap
            valid.append(ap)
    per_label["macro"] = float(np.mean(valid)) if valid else 0.0
    return per_label


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den > 0 else 0.0


def confusion_at_threshold(
    y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5
):
    """Per-label (TP, FP, TN, FN) count vectors after thresholding scores."""
    y_pred = (y_score >= threshold).astype(int)
    yt = y_true.astype(int)
    tp = ((y_pred == 1) & (yt == 1)).sum(axis=0)
    fp = ((y_pred == 1) & (yt == 0)).sum(axis=0)
    tn = ((y_pred == 0) & (yt == 0)).sum(axis=0)
    fn = ((y_pred == 0) & (yt == 1)).sum(axis=0)
    return tp, fp, tn, fn


def operating_point_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float = 0.5,
    labels: list[str] | None = None,
) -> dict:
    """Sensitivity, specificity, PPV, NPV at the decision threshold.

    These are the numbers a clinician reads off a model: sensitivity (recall) and
    specificity trade off false negatives against false positives, while PPV/NPV
    (precision and its negative counterpart) state how much a positive/negative
    call can be trusted at the test-set prevalence. Reported per label with the
    positive support count, plus a macro average over labels where each metric is
    defined (its denominator is non-zero).
    """
    tp, fp, tn, fn = confusion_at_threshold(y_true, y_score, threshold)
    per_label: dict[str, dict] = {}
    sens_vals, spec_vals, ppv_vals, npv_vals = [], [], [], []
    for c in range(y_true.shape[1]):
        pos, neg = tp[c] + fn[c], tn[c] + fp[c]
        sens = _safe_div(tp[c], pos)
        spec = _safe_div(tn[c], neg)
        ppv = _safe_div(tp[c], tp[c] + fp[c])
        npv = _safe_div(tn[c], tn[c] + fn[c])
        name = labels[c] if labels is not None and c < len(labels) else str(c)
        per_label[name] = {
            "sensitivity": sens,
            "specificity": spec,
            "ppv": ppv,
            "npv": npv,
            "support_pos": int(pos),
            "support_neg": int(neg),
        }
        if pos > 0:
            sens_vals.append(sens)
            ppv_vals.append(ppv)
        if neg > 0:
            spec_vals.append(spec)
            npv_vals.append(npv)

    def _macro(vals: list[float]) -> float:
        return float(np.mean(vals)) if vals else 0.0

    return {
        "threshold": threshold,
        "macro": {
            "sensitivity": _macro(sens_vals),
            "specificity": _macro(spec_vals),
            "ppv": _macro(ppv_vals),
            "npv": _macro(npv_vals),
        },
        "per_label": per_label,
    }


def brier_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Mean squared error between predicted probabilities and labels.

    Lower is better (0 = perfect). Averaged over every label of every example,
    it is a proper scoring rule that rewards both discrimination and calibration.
    """
    return float(np.mean((y_score.astype(float) - y_true.astype(float)) ** 2))


def expected_calibration_error(
    y_true: np.ndarray, y_score: np.ndarray, n_bins: int = 10
) -> float:
    """Expected Calibration Error over all label probabilities.

    Bins predicted probabilities into ``n_bins`` equal-width bins and returns the
    support-weighted mean gap between confidence (mean predicted probability) and
    accuracy (observed positive rate) in each bin. 0 means a probability of p is
    right p of the time.
    """
    p = np.asarray(y_score, dtype=float).ravel()
    y = np.asarray(y_true, dtype=float).ravel()
    n = p.size
    if n == 0:
        return 0.0
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (p > lo) & (p <= hi) if i > 0 else (p >= lo) & (p <= hi)
        count = int(mask.sum())
        if count:
            ece += (count / n) * abs(y[mask].mean() - p[mask].mean())
    return float(ece)


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
    macro_sens_samples: list[float] = []
    macro_spec_samples: list[float] = []
    brier_samples: list[float] = []
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
        op = operating_point_metrics(yt, ys, threshold)["macro"]
        macro_sens_samples.append(op["sensitivity"])
        macro_spec_samples.append(op["specificity"])
        brier_samples.append(brier_score(yt, ys))

    return {
        "n_boot": n_boot,
        "ci_level": ci,
        "seed": seed,
        "macro_auroc": _percentile_ci(macro_auroc_samples, ci),
        "macro_f1": _percentile_ci(macro_f1_samples, ci),
        "macro_sensitivity": _percentile_ci(macro_sens_samples, ci),
        "macro_specificity": _percentile_ci(macro_spec_samples, ci),
        "brier": _percentile_ci(brier_samples, ci),
        "per_label_auroc": {
            key: _percentile_ci(vals, ci) for key, vals in per_label_samples.items()
        },
    }
