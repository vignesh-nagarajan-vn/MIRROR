"""Ablation: classification-only baseline vs. the full MIRROR pipeline.

MIRROR's research question is comparative — does adding the localisation and
reasoning layers help interpretability *without* degrading prediction? Answering
it needs the baseline the question names: a classification-only condition to
compare against. This harness defines three conditions and assembles the
side-by-side table the paper's results section needs:

    classification_only   predictions only (where most imaging models stop)
    with_localization      predictions + Grad-CAM/Score-CAM evidence
    full_mirror            predictions + localisation + grounded NL report

Two things make the comparison honest:

1. **No predictive cost.** Layers 2 and 3 are post-hoc, so the predictions are
   identical across conditions. The harness verifies this empirically on the
   sample (``predictions_invariant``) — the AUROC/F1 column is therefore the same
   in every row, which is exactly the point.
2. **Explicit added capability and its latency.** Each row records which
   capabilities it unlocks (evidence localisation, NL report) and the per-stage
   wall-clock cost of getting them, profiled on real images.

Predictive numbers (AUROC/F1) and localisation numbers (pointing game / IoU) are
read from the JSON written by ``evaluate.py`` and ``evaluate_localization.py`` so
nothing is recomputed; pass them with ``--prediction-results`` /
``--localization-results``. The capability matrix and latency profile run on the
bundled synthetic samples with no downloads, so the table is partially
reproducible out of the box.

Usage::

    python -m evaluation.ablation --config configs/default.yaml \
        --prediction-results evaluation/results/eval_densenet121.json \
        --localization-results evaluation/results/loc_densenet121_gradcam.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

try:
    import torch  # noqa: F401 - only needed for the live latency profile
except ImportError:  # pragma: no cover
    torch = None

from models.common.config import Config

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".dcm")


@dataclass(frozen=True)
class Condition:
    """One ablation condition: which post-hoc layers are switched on."""

    name: str
    localize: bool
    report: bool
    description: str


# The three conditions, in increasing capability. ``full_mirror`` is the default
# pipeline behaviour; ``classification_only`` is the baseline the research
# question compares against.
CONDITIONS: tuple[Condition, ...] = (
    Condition(
        "classification_only",
        localize=False,
        report=False,
        description="Predictions only — where most medical-imaging models stop.",
    ),
    Condition(
        "with_localization",
        localize=True,
        report=False,
        description="Predictions + Grad-CAM/Score-CAM evidence localisation.",
    ),
    Condition(
        "full_mirror",
        localize=True,
        report=True,
        description="Predictions + localisation + grounded natural-language report.",
    ),
)


def capability_matrix(conditions: tuple[Condition, ...] = CONDITIONS) -> dict[str, dict]:
    """Which capabilities each condition provides (prediction is always on)."""
    return {
        c.name: {
            "prediction": True,
            "localization": c.localize,
            "report": c.report,
        }
        for c in conditions
    }


def merge_metric_results(
    prediction_results: dict | None, localization_results: dict | None
) -> dict:
    """Pull the headline numbers out of the two evaluation JSON summaries."""
    pred = prediction_results or {}
    loc = (localization_results or {}).get("overall", {})
    op = pred.get("operating_point", {}).get("macro", {})
    return {
        "macro_auroc": pred.get("macro_auroc"),
        "macro_auprc": pred.get("macro_auprc"),
        "macro_f1": pred.get("macro_f1"),
        "sensitivity": op.get("sensitivity"),
        "specificity": op.get("specificity"),
        "pointing_game": loc.get("pointing_game"),
        "mean_iou": loc.get("mean_iou"),
        "loc_accuracy": loc.get("loc_accuracy"),
    }


def assemble_table(
    metrics: dict,
    latency: dict | None = None,
    conditions: tuple[Condition, ...] = CONDITIONS,
) -> list[dict]:
    """Build the comparison table: one row per condition.

    Predictive metrics apply to every row (predictions are condition-invariant);
    localisation metrics only populate rows whose localisation layer is on; the
    report column reflects whether the reasoning layer is on. ``latency`` maps a
    condition name to its per-stage timing dict (or is None when not profiled).
    """
    rows: list[dict] = []
    for c in conditions:
        rows.append(
            {
                "condition": c.name,
                "description": c.description,
                # Capabilities.
                "prediction": True,
                "localization": c.localize,
                "report": c.report,
                # Predictive quality — same across conditions by construction.
                "macro_auroc": metrics.get("macro_auroc"),
                "macro_auprc": metrics.get("macro_auprc"),
                "macro_f1": metrics.get("macro_f1"),
                "sensitivity": metrics.get("sensitivity"),
                "specificity": metrics.get("specificity"),
                # Explanation quality — only where the localisation layer is on.
                "pointing_game": metrics.get("pointing_game") if c.localize else None,
                "mean_iou": metrics.get("mean_iou") if c.localize else None,
                "loc_accuracy": metrics.get("loc_accuracy") if c.localize else None,
                # Cost.
                "latency_ms": (latency or {}).get(c.name),
            }
        )
    return rows


def find_images(images_arg: str, limit: int | None) -> list[Path]:
    """Resolve a directory or glob into a sorted list of image paths."""
    p = Path(images_arg)
    if p.is_dir():
        paths = sorted(
            f for f in p.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS
        )
    else:
        paths = sorted(Path().glob(images_arg))
    return paths[:limit] if limit else paths


def profile_conditions(pipeline, images: list[Path], conditions=CONDITIONS) -> dict:
    """Run each condition over the sample and measure latency + invariance.

    Returns ``{"latency": {condition: {stage: mean_ms,...}}, "predictions_invariant":
    bool, "max_prob_delta": float, "n_images": int}``. The invariance check
    confirms the later layers do not perturb the predictions.
    """
    import numpy as np

    per_stage: dict[str, dict[str, list[float]]] = {c.name: {} for c in conditions}
    # Predictions of the first ("classification_only") condition are the reference.
    reference: dict[str, list[float]] = {}
    max_delta = 0.0

    for img in images:
        raw = img.read_bytes()
        per_image_probs: dict[str, dict[str, float]] = {}
        for c in conditions:
            result = pipeline.analyze(raw, localize=c.localize, report=c.report)
            for stage, ms in result.meta["timings_ms"].items():
                per_stage[c.name].setdefault(stage, []).append(ms)
            per_image_probs[c.name] = {
                f.label: f.probability for f in result.findings
            }

        ref = per_image_probs[conditions[0].name]
        for c in conditions[1:]:
            for label, prob in per_image_probs[c.name].items():
                max_delta = max(max_delta, abs(prob - ref.get(label, prob)))

    latency = {
        name: {stage: float(np.mean(vals)) for stage, vals in stages.items()}
        for name, stages in per_stage.items()
    }
    for name, stages in latency.items():
        stages["total"] = float(sum(stages.values()))

    return {
        "latency": latency,
        "predictions_invariant": max_delta < 1e-6,
        "max_prob_delta": float(max_delta),
        "n_images": len(images),
    }


def _load_json(path: str | None) -> dict | None:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"Results file not found: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def _fmt(v) -> str:
    # ASCII only — the table prints to terminals (e.g. Windows cp1252) that
    # cannot encode box-drawing/check glyphs.
    if v is None:
        return "-"
    if isinstance(v, bool):
        return "yes" if v else "no"
    return f"{v:.4f}"


def print_table(rows: list[dict]) -> None:
    """Pretty-print the ablation table to stdout."""
    cols = [
        ("condition", "Condition", 20),
        ("localization", "Local.", 7),
        ("report", "Report", 7),
        ("macro_auroc", "AUROC", 8),
        ("macro_f1", "F1", 8),
        ("pointing_game", "Point", 8),
        ("mean_iou", "IoU", 8),
    ]
    header = " ".join(f"{title:<{w}}" for _, title, w in cols)
    print(header)
    print("-" * len(header))
    for r in rows:
        cells = []
        for key, _, w in cols:
            val = r[key] if key == "condition" else _fmt(r[key])
            cells.append(f"{val:<{w}}")
        print(" ".join(cells))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ablation: classification-only vs. full MIRROR."
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--images",
        default="datasets/samples/chestxray14/images",
        help="Directory or glob of images to profile latency on.",
    )
    parser.add_argument("--n", type=int, default=8, help="Max images to profile.")
    parser.add_argument("--prediction-results", default=None)
    parser.add_argument("--localization-results", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no-latency",
        action="store_true",
        help="Skip the live latency/invariance profile (no model needed).",
    )
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    if args.checkpoint:
        config.model.checkpoint_path = args.checkpoint

    pred_json = _load_json(args.prediction_results)
    loc_json = _load_json(args.localization_results)
    metrics = merge_metric_results(pred_json, loc_json)

    profile = None
    if not args.no_latency:
        if torch is None:
            raise SystemExit(
                "PyTorch is required for the latency profile; pass --no-latency "
                "to assemble the capability/metrics table without it."
            )
        import numpy as np

        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        from models.pipeline import MirrorPipeline

        images = find_images(args.images, args.n)
        if not images:
            raise SystemExit(f"No images found under {args.images}")
        pipeline = MirrorPipeline(config)
        profile = profile_conditions(pipeline, images)

    latency = profile["latency"] if profile else None
    rows = assemble_table(metrics, latency)

    summary = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "backbone": config.model.backbone,
        "explain_method": config.explain.method,
        "report_provider": config.report.provider,
        "capabilities": capability_matrix(),
        "metrics_source": {
            "prediction_results": args.prediction_results,
            "localization_results": args.localization_results,
        },
        "profile": profile,
        "table": rows,
    }

    out_dir = Path("evaluation/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"ablation_{config.model.backbone}.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print_table(rows)
    if profile:
        inv = "yes" if profile["predictions_invariant"] else "NO (!)"
        print(
            f"\npredictions invariant across conditions: {inv} "
            f"(max prob delta = {profile['max_prob_delta']:.2e}, "
            f"n={profile['n_images']} images)"
        )
        for name, stages in profile["latency"].items():
            print(f"  {name:<20} total {stages['total']:8.1f} ms/img")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
