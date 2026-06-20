"""Tests for multi-seed aggregation (``evaluation/aggregate_seeds``).

Torch-free: checks the mean/std math, seed extraction, and per-label handling
over synthetic per-seed summaries shaped like ``evaluate.py`` output.

Run:  pytest tests/test_aggregate_seeds.py
"""

from __future__ import annotations

import pytest

from evaluation.aggregate_seeds import aggregate


def _summary(seed, auroc, f1, per_label):
    return {
        "backbone": "densenet121",
        "macro_auroc": auroc,
        "macro_f1": f1,
        "per_label_auroc": per_label,
        "reproducibility": {"seed": seed},
    }


def test_aggregate_mean_and_std():
    summaries = [
        _summary(0, 0.80, 0.40, {"Mass": 0.7}),
        _summary(1, 0.82, 0.42, {"Mass": 0.9}),
        _summary(2, 0.84, 0.44, {"Mass": 0.8}),
    ]
    out = aggregate(summaries)
    assert out["n_seeds"] == 3
    assert out["seeds"] == [0, 1, 2]
    assert out["backbone"] == "densenet121"
    assert out["macro_auroc"]["mean"] == pytest.approx(0.82)
    # sample std (ddof=1) of [0.80, 0.82, 0.84] = 0.02
    assert out["macro_auroc"]["std"] == pytest.approx(0.02)
    assert out["macro_f1"]["mean"] == pytest.approx(0.42)
    assert out["per_label_auroc"]["Mass"]["mean"] == pytest.approx(0.8)


def test_aggregate_single_seed_zero_std():
    out = aggregate([_summary(0, 0.81, 0.41, {"Nodule": 0.6})])
    assert out["n_seeds"] == 1
    assert out["macro_auroc"]["std"] == 0.0
    assert out["per_label_auroc"]["Nodule"]["n"] == 1


def test_aggregate_falls_back_to_top_level_seed():
    s = {"backbone": "vit_b_16", "macro_auroc": 0.7, "macro_f1": 0.3,
         "per_label_auroc": {}, "seed": 5}
    out = aggregate([s])
    assert out["seeds"] == [5]


def test_aggregate_handles_partial_per_label():
    # "Hernia" appears in only one run; it is averaged over the runs that have it.
    summaries = [
        _summary(0, 0.80, 0.40, {"Mass": 0.7, "Hernia": 0.95}),
        _summary(1, 0.82, 0.42, {"Mass": 0.9}),
    ]
    out = aggregate(summaries)
    assert out["per_label_auroc"]["Mass"]["n"] == 2
    assert out["per_label_auroc"]["Hernia"]["n"] == 1
    assert out["per_label_auroc"]["Hernia"]["mean"] == pytest.approx(0.95)


def test_aggregate_mixed_backbones_listed():
    summaries = [
        _summary(0, 0.80, 0.40, {}),
        {**_summary(1, 0.82, 0.42, {}), "backbone": "efficientnet_b0"},
    ]
    out = aggregate(summaries)
    assert out["backbone"] == ["densenet121", "efficientnet_b0"]


def test_aggregate_empty_raises():
    with pytest.raises(ValueError):
        aggregate([])
