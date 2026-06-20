"""Tests for bootstrap confidence intervals (``evaluation/metrics.bootstrap_cis``).

Torch-free: needs only numpy + scikit-learn. They check the structure of the CI
output, determinism under a fixed seed, ordering (lo <= mean <= hi), and the
degenerate case of perfect separation (CI collapses to 1.0).

Run:  pytest tests/test_metrics_bootstrap.py
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("sklearn")

from evaluation.metrics import bootstrap_cis  # noqa: E402


def _separable_data(n: int = 200, c: int = 3, seed: int = 0):
    """Two-class-per-label data where score perfectly ranks the positives."""
    rng = np.random.default_rng(seed)
    y_true = rng.integers(0, 2, size=(n, c)).astype(float)
    # Scores equal to labels plus a tiny margin -> perfect separation.
    y_score = y_true * 0.9 + 0.05
    return y_true, y_score


def test_bootstrap_cis_structure():
    y_true, y_score = _separable_data()
    out = bootstrap_cis(y_true, y_score, n_boot=50, ci=0.95, seed=1)
    assert out["n_boot"] == 50
    assert out["ci_level"] == 0.95
    for block in (out["macro_auroc"], out["macro_f1"]):
        assert set(block) == {"mean", "std", "lo", "hi"}
    assert isinstance(out["per_label_auroc"], dict)


def test_bootstrap_cis_is_deterministic_under_seed():
    y_true, y_score = _separable_data()
    a = bootstrap_cis(y_true, y_score, n_boot=100, seed=7)
    b = bootstrap_cis(y_true, y_score, n_boot=100, seed=7)
    assert a["macro_auroc"] == b["macro_auroc"]
    assert a["macro_f1"] == b["macro_f1"]


def test_bootstrap_cis_interval_ordering():
    rng = np.random.default_rng(3)
    y_true = rng.integers(0, 2, size=(300, 4)).astype(float)
    y_score = 0.5 * y_true + 0.5 * rng.random((300, 4))  # noisy but informative
    out = bootstrap_cis(y_true, y_score, n_boot=200, ci=0.9, seed=5)
    for block in (out["macro_auroc"], out["macro_f1"]):
        assert block["lo"] <= block["mean"] <= block["hi"]
        assert 0.0 <= block["lo"] <= 1.0
        assert 0.0 <= block["hi"] <= 1.0


def test_bootstrap_perfect_separation_collapses_to_one():
    y_true, y_score = _separable_data(seed=2)
    out = bootstrap_cis(y_true, y_score, n_boot=100, seed=11)
    assert out["macro_auroc"]["lo"] == pytest.approx(1.0)
    assert out["macro_auroc"]["hi"] == pytest.approx(1.0)
    assert out["macro_auroc"]["std"] == pytest.approx(0.0)


def test_wider_ci_is_not_narrower():
    rng = np.random.default_rng(9)
    y_true = rng.integers(0, 2, size=(250, 3)).astype(float)
    y_score = 0.4 * y_true + 0.6 * rng.random((250, 3))
    narrow = bootstrap_cis(y_true, y_score, n_boot=300, ci=0.80, seed=4)["macro_auroc"]
    wide = bootstrap_cis(y_true, y_score, n_boot=300, ci=0.99, seed=4)["macro_auroc"]
    assert (wide["hi"] - wide["lo"]) >= (narrow["hi"] - narrow["lo"])
