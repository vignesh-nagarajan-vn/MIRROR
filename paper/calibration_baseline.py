"""Macro Brier score of a no-skill, constant-prevalence predictor on ChestMNIST.

Arithmetic on already-committed data, not a new experiment. For each of the 14
labels the evaluation JSON records ``support_pos`` and ``support_neg`` at the
operating point, so the label prevalence is

    p = support_pos / (support_pos + support_neg)

A predictor that ignores the image and always emits ``p`` for that label scores a
Brier score of exactly ``p * (1 - p)`` on it (the variance of a Bernoulli(p)).
Averaging over the 14 labels gives the macro Brier of the no-skill baseline, the
number MIRROR's measured macro Brier has to be read against.

Run:
    python paper/calibration_baseline.py
"""

from __future__ import annotations

import json
from pathlib import Path

EVAL_JSON = (
    Path(__file__).resolve().parents[1]
    / "results"
    / "chestmnist"
    / "eval_chestmnist_densenet121.json"
)


def main() -> None:
    data = json.loads(EVAL_JSON.read_text(encoding="utf-8"))
    per_label = data["operating_point"]["per_label"]

    rows = []
    for label, m in per_label.items():
        pos, neg = m["support_pos"], m["support_neg"]
        prev = pos / (pos + neg)
        rows.append((label, pos, neg, prev, prev * (1.0 - prev), m["sensitivity"]))

    baseline = sum(r[4] for r in rows) / len(rows)
    model = data["calibration"]["brier"]
    silent = sum(1 for r in rows if r[5] == 0.0)

    width = max(len(r[0]) for r in rows)
    print(f"{'label':<{width}}  {'pos':>5} {'neg':>6} {'prev':>8} {'p(1-p)':>8} {'sens':>7}")
    for label, pos, neg, prev, brier, sens in rows:
        print(f"{label:<{width}}  {pos:>5} {neg:>6} {prev:>8.5f} {brier:>8.5f} {sens:>7.4f}")

    print()
    print(f"n_test                                 = {data['n_test']}")
    print(f"labels with sensitivity exactly 0.0    = {silent} of {len(rows)}")
    print(f"macro Brier, constant-prevalence model = {baseline:.6f}")
    print(f"macro Brier, MIRROR DenseNet-121       = {model:.6f}")
    print(f"absolute difference                    = {baseline - model:.6f}")
    print(f"model Brier as a fraction of baseline  = {model / baseline:.4f}")
    print(f"macro ECE, MIRROR DenseNet-121         = {data['calibration']['ece']:.6f}")
    print(f"macro F1, MIRROR DenseNet-121          = {data['macro_f1']:.6f}")


if __name__ == "__main__":
    main()
