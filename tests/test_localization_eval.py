"""Tests for the localization evaluation harness (``evaluation/evaluate_localization``).

These cover the pure, torch-free pieces: box conversion/scaling, the per-box
scoring against synthetic heatmaps, and per-label aggregation. They need only
numpy, so they run without a trained model or the NIH box file.

Run:  pytest tests/test_localization_eval.py
"""

from __future__ import annotations

import numpy as np
import pytest

from evaluation.evaluate_localization import (
    BBOX_LABEL_ALIASES,
    scale_box,
    score_box,
    to_xyxy,
    _aggregate,
)


def test_to_xyxy_converts_corners():
    assert to_xyxy(10, 20, 30, 40) == (10, 20, 40, 60)


def test_scale_box_downscales_and_clips():
    # 1024px original box -> 224px heatmap grid.
    box = to_xyxy(512, 512, 256, 256)  # (512, 512, 768, 768)
    x0, y0, x1, y1 = scale_box(box, orig_size=(1024, 1024), cam_size=(224, 224))
    assert (x0, y0) == (112, 112)
    assert (x1, y1) == (168, 168)


def test_scale_box_stays_in_bounds():
    # A box flush against the original edge must not exceed the heatmap grid.
    box = to_xyxy(1000, 1000, 100, 100)
    x0, y0, x1, y1 = scale_box(box, orig_size=(1024, 1024), cam_size=(224, 224))
    assert 0 <= x0 < x1 <= 224
    assert 0 <= y0 < y1 <= 224


def test_scale_box_always_has_positive_area():
    # A degenerate sub-pixel box must still yield a >=1px slice-able region.
    box = to_xyxy(500, 500, 1, 1)
    x0, y0, x1, y1 = scale_box(box, orig_size=(1024, 1024), cam_size=(224, 224))
    assert x1 > x0 and y1 > y0


def _cam_with_peak_at(shape, peak_xy, hot_box=None):
    """A heatmap that is 1.0 inside ``hot_box`` (or at ``peak_xy``), else 0."""
    cam = np.zeros(shape, dtype=float)
    if hot_box is not None:
        x0, y0, x1, y1 = hot_box
        cam[y0:y1, x0:x1] = 1.0
    px, py = peak_xy
    cam[py, px] = 1.0
    return cam


def test_score_box_pointing_hit_inside():
    cam = _cam_with_peak_at((100, 100), peak_xy=(50, 50))
    out = score_box(cam, gt_box_xyxy=(40, 40, 60, 60))
    assert out["pointing_hit"] is True


def test_score_box_pointing_miss_outside():
    cam = _cam_with_peak_at((100, 100), peak_xy=(10, 10))
    out = score_box(cam, gt_box_xyxy=(40, 40, 60, 60))
    assert out["pointing_hit"] is False


def test_score_box_perfect_overlap_iou_one():
    box = (30, 30, 70, 70)
    cam = _cam_with_peak_at((100, 100), peak_xy=(50, 50), hot_box=box)
    out = score_box(cam, gt_box_xyxy=box, cam_threshold=0.5)
    assert out["iou"] == pytest.approx(1.0)
    assert out["loc_correct"] is True


def test_score_box_loc_correct_respects_threshold():
    # Hot region is a quarter of the GT box -> IoU = 0.25.
    cam = _cam_with_peak_at((100, 100), peak_xy=(35, 35), hot_box=(30, 30, 50, 50))
    out = score_box(cam, gt_box_xyxy=(30, 30, 70, 70), cam_threshold=0.5, iou_threshold=0.3)
    assert out["iou"] == pytest.approx(0.25)
    assert out["loc_correct"] is False  # 0.25 < 0.30


def test_aggregate_means():
    records = [
        {"pointing_hit": True, "iou": 0.4, "loc_correct": True},
        {"pointing_hit": False, "iou": 0.2, "loc_correct": False},
    ]
    agg = _aggregate(records)
    assert agg["n"] == 2
    assert agg["pointing_game"] == pytest.approx(0.5)
    assert agg["mean_iou"] == pytest.approx(0.3)
    assert agg["loc_accuracy"] == pytest.approx(0.5)


def test_aggregate_empty_is_zeroed():
    agg = _aggregate([])
    assert agg == {"n": 0, "pointing_game": 0.0, "mean_iou": 0.0, "loc_accuracy": 0.0}


def test_infiltrate_alias_maps_to_canonical_label():
    assert BBOX_LABEL_ALIASES["Infiltrate"] == "Infiltration"
