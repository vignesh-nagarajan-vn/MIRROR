"""Aggregate evaluation results across training seeds.

A single training run reflects one draw of the optimisation noise. To report a
result that is robust to that noise — what reviewers expect on ChestX-ray14 — you
train the model under several seeds, evaluate each checkpoint, and summarise the
metrics as mean +/- standard deviation across seeds.

Workflow::

    for s in 0 1 2; do
        python -m models.classification.train --config configs/default.yaml --seed $s
        python -m evaluation.evaluate --config configs/default.yaml \
            --checkpoint models/checkpoints/densenet121_best.pt
        cp evaluation/results/eval_densenet121.json evaluation/results/eval_seed$s.json
    done
    python -m evaluation.aggregate_seeds evaluation/results/eval_seed*.json

This complements ``evaluate.py``'s bootstrap CIs: bootstrap quantifies
test-set sampling uncertainty for one model; this quantifies training-seed
uncertainty across models. A paper ideally reports both.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np


def _mean_std(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "n": int(arr.size),
    }


def aggregate(summaries: list[dict]) -> dict:
    """Summarise macro AUROC/F1 and per-label AUROC across seed result dicts.

    Each input is a summary written by ``evaluate.py``. Seeds are read from each
    summary's ``reproducibility.seed`` (falling back to a top-level ``seed``).
    Per-label labels present in only some runs are averaged over the runs that
    have them.
    """
    if not summaries:
        raise ValueError("No result summaries to aggregate.")

    seeds = []
    for s in summaries:
        seed = s.get("reproducibility", {}).get("seed", s.get("seed"))
        seeds.append(seed)

    macro_auroc = _mean_std([s["macro_auroc"] for s in summaries])
    macro_f1 = _mean_std([s["macro_f1"] for s in summaries])

    per_label: dict[str, list[float]] = {}
    for s in summaries:
        for label, value in s.get("per_label_auroc", {}).items():
            per_label.setdefault(label, []).append(value)

    # Aggregate the clinical macro metrics too, when the runs carry them (newer
    # eval_*.json). Each is extracted by a path so nested blocks work; a metric is
    # aggregated over the runs that actually report it.
    optional_paths = {
        "macro_auprc": ("macro_auprc",),
        "macro_sensitivity": ("operating_point", "macro", "sensitivity"),
        "macro_specificity": ("operating_point", "macro", "specificity"),
        "macro_ppv": ("operating_point", "macro", "ppv"),
        "macro_npv": ("operating_point", "macro", "npv"),
        "brier": ("calibration", "brier"),
        "ece": ("calibration", "ece"),
    }

    def _dig(d: dict, path: tuple[str, ...]):
        for key in path:
            if not isinstance(d, dict) or key not in d:
                return None
            d = d[key]
        return d

    optional: dict[str, dict] = {}
    for name, path in optional_paths.items():
        vals = [v for s in summaries if (v := _dig(s, path)) is not None]
        if vals:
            optional[name] = _mean_std(vals)

    backbones = sorted({s.get("backbone") for s in summaries if s.get("backbone")})
    modalities = sorted({s.get("modality") for s in summaries if s.get("modality")})
    return {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "n_seeds": len(summaries),
        "seeds": seeds,
        "backbone": backbones[0] if len(backbones) == 1 else backbones,
        "modality": modalities[0] if len(modalities) == 1 else modalities,
        "macro_auroc": macro_auroc,
        "macro_f1": macro_f1,
        **optional,
        "per_label_auroc": {k: _mean_std(v) for k, v in sorted(per_label.items())},
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate per-seed evaluation JSON into mean +/- std."
    )
    parser.add_argument("results", nargs="+", help="eval_*.json files, one per seed.")
    parser.add_argument(
        "--out",
        default=None,
        help="Output path (default: evaluation/results/aggregate_<backbone>.json).",
    )
    args = parser.parse_args()

    summaries = [json.loads(Path(p).read_text(encoding="utf-8")) for p in args.results]
    result = aggregate(summaries)

    out_dir = Path("evaluation/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    backbone = result["backbone"]
    tag = backbone if isinstance(backbone, str) else "mixed"
    out_path = Path(args.out) if args.out else out_dir / f"aggregate_{tag}.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    a, f = result["macro_auroc"], result["macro_f1"]
    print(
        f"across {result['n_seeds']} seeds {result['seeds']}: "
        f"macro AUROC = {a['mean']:.4f} +/- {a['std']:.4f}  |  "
        f"macro F1 = {f['mean']:.4f} +/- {f['std']:.4f}"
    )
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
