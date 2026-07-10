"""Tests for the clinical-grade metrics (``evaluation/metrics.py``).

These cover the pure-NumPy metrics a diagnostic model is judged by at its
operating point (sensitivity/specificity/PPV/NPV) and its calibration (Brier,
ECE). They deliberately need only NumPy so they run without scikit-learn or the
ML stack, and they check against hand-computed values on tiny fixtures.

Run:  pytest tests/test_clinical_metrics.py
"""

from __future__ import annotations

import numpy as np
import pytest

from evaluation.metrics import (
    brier_score,
    confusion_at_threshold,
    expected_calibration_error,
    operating_point_metrics,
)

# A 4-example, 2-label fixture. Label A is imperfect, label B is perfect.
Y_TRUE = np.array([[1, 0], [1, 0], [0, 1], [0, 1]])
Y_SCORE = np.array([[0.9, 0.2], [0.4, 0.1], [0.3, 0.8], [0.6, 0.9]])


def test_confusion_counts():
    tp, fp, tn, fn = confusion_at_threshold(Y_TRUE, Y_SCORE, 0.5)
    # Label A: preds [1,0,0,1] vs true [1,1,0,0] -> TP=1, FP=1, TN=1, FN=1.
    assert (tp[0], fp[0], tn[0], fn[0]) == (1, 1, 1, 1)
    # Label B: preds [0,0,1,1] vs true [0,0,1,1] -> perfect.
    assert (tp[1], fp[1], tn[1], fn[1]) == (2, 0, 2, 0)


def test_operating_point_per_label_and_macro():
    op = operating_point_metrics(Y_TRUE, Y_SCORE, 0.5, labels=["A", "B"])
    a = op["per_label"]["A"]
    assert a == {
        "sensitivity": 0.5,
        "specificity": 0.5,
        "ppv": 0.5,
        "npv": 0.5,
        "support_pos": 2,
        "support_neg": 2,
    }
    b = op["per_label"]["B"]
    assert (b["sensitivity"], b["specificity"], b["ppv"], b["npv"]) == (1.0, 1.0, 1.0, 1.0)
    # Macro is the mean over defined labels.
    assert op["macro"]["sensitivity"] == 0.75
    assert op["macro"]["specificity"] == 0.75


def test_operating_point_macro_skips_undefined_labels():
    # A label with no positives has undefined sensitivity; it must be excluded
    # from the macro sensitivity rather than counted as zero.
    y_true = np.array([[1, 0], [0, 0]])
    y_score = np.array([[0.9, 0.1], [0.2, 0.1]])
    op = operating_point_metrics(y_true, y_score, 0.5)
    # Only label 0 has a positive -> its sensitivity (1.0) is the macro.
    assert op["macro"]["sensitivity"] == 1.0
    assert op["per_label"]["1"]["support_pos"] == 0


def test_brier_score_matches_hand_value():
    assert brier_score(Y_TRUE, Y_SCORE) == pytest.approx(0.115)


def test_brier_zero_for_perfect_confident_predictions():
    y = np.array([[1, 0], [0, 1]])
    assert brier_score(y, y.astype(float)) == 0.0


def test_ece_zero_when_perfectly_calibrated():
    # Bin probabilities where predicted confidence equals observed frequency.
    y_true = np.array([[1, 0, 1, 0]]).reshape(4, 1)
    y_score = np.array([[1.0, 0.0, 1.0, 0.0]]).reshape(4, 1)
    assert expected_calibration_error(y_true, y_score, n_bins=10) == 0.0


def test_ece_detects_miscalibration():
    # Always predict 0.9 but only right half the time -> gap ~0.4.
    y_true = np.array([[1], [0], [1], [0]])
    y_score = np.full((4, 1), 0.9)
    ece = expected_calibration_error(y_true, y_score, n_bins=10)
    assert abs(ece - 0.4) < 1e-9
