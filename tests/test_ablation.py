"""Tests for the ablation harness (``evaluation/ablation``).

These cover the torch-free logic: the condition definitions, the capability
matrix, merging the two evaluation JSON summaries, and assembling the comparison
table. The live latency/invariance profile (which needs a model) is not exercised
here.

Run:  pytest tests/test_ablation.py
"""

from __future__ import annotations

from evaluation.ablation import (
    CONDITIONS,
    assemble_table,
    capability_matrix,
    merge_metric_results,
)


def test_conditions_span_baseline_to_full():
    names = [c.name for c in CONDITIONS]
    assert names == ["classification_only", "with_localization", "full_mirror"]
    baseline, mid, full = CONDITIONS
    assert (baseline.localize, baseline.report) == (False, False)
    assert (mid.localize, mid.report) == (True, False)
    assert (full.localize, full.report) == (True, True)


def test_capability_matrix_prediction_always_on():
    caps = capability_matrix()
    assert all(c["prediction"] for c in caps.values())
    assert caps["classification_only"] == {
        "prediction": True,
        "localization": False,
        "report": False,
    }
    assert caps["full_mirror"]["report"] is True


def test_merge_metric_results_pulls_headline_numbers():
    pred = {"macro_auroc": 0.82, "macro_f1": 0.41, "extra": "ignored"}
    loc = {"overall": {"pointing_game": 0.7, "mean_iou": 0.25, "loc_accuracy": 0.6}}
    merged = merge_metric_results(pred, loc)
    assert merged["macro_auroc"] == 0.82
    assert merged["macro_f1"] == 0.41
    assert merged["pointing_game"] == 0.7
    assert merged["mean_iou"] == 0.25
    assert merged["loc_accuracy"] == 0.6


def test_merge_metric_results_tolerates_missing():
    merged = merge_metric_results(None, None)
    assert set(merged) == {
        "macro_auroc",
        "macro_f1",
        "pointing_game",
        "mean_iou",
        "loc_accuracy",
    }
    assert all(v is None for v in merged.values())


def test_assemble_table_prediction_metrics_apply_to_all_rows():
    metrics = {
        "macro_auroc": 0.82,
        "macro_f1": 0.41,
        "pointing_game": 0.7,
        "mean_iou": 0.25,
        "loc_accuracy": 0.6,
    }
    rows = assemble_table(metrics)
    assert len(rows) == 3
    # AUROC/F1 are identical across every condition (post-hoc layers).
    assert {r["macro_auroc"] for r in rows} == {0.82}
    assert {r["macro_f1"] for r in rows} == {0.41}


def test_assemble_table_localization_only_where_enabled():
    metrics = {
        "macro_auroc": 0.82,
        "macro_f1": 0.41,
        "pointing_game": 0.7,
        "mean_iou": 0.25,
        "loc_accuracy": 0.6,
    }
    rows = {r["condition"]: r for r in assemble_table(metrics)}
    # Classification-only: no localisation metrics.
    base = rows["classification_only"]
    assert base["pointing_game"] is None and base["mean_iou"] is None
    assert base["localization"] is False and base["report"] is False
    # With localisation: metrics present, still no report.
    loc = rows["with_localization"]
    assert loc["pointing_game"] == 0.7 and loc["mean_iou"] == 0.25
    assert loc["report"] is False
    # Full MIRROR: everything on.
    full = rows["full_mirror"]
    assert full["pointing_game"] == 0.7 and full["report"] is True


def test_assemble_table_latency_attached_per_condition():
    latency = {
        "classification_only": {"prediction": 10.0, "total": 10.0},
        "with_localization": {"prediction": 10.0, "localization": 40.0, "total": 50.0},
        "full_mirror": {"prediction": 10.0, "report": 5.0, "total": 55.0},
    }
    rows = {r["condition"]: r for r in assemble_table({}, latency=latency)}
    assert rows["classification_only"]["latency_ms"]["total"] == 10.0
    assert rows["full_mirror"]["latency_ms"]["total"] == 55.0


def test_assemble_table_without_latency_is_none():
    rows = assemble_table({"macro_auroc": 0.8})
    assert all(r["latency_ms"] is None for r in rows)
