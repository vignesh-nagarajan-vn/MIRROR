"""Evaluate MIRROR's *explanations* against ground-truth pathology boxes.

This is the experiment behind MIRROR's central novelty claim — that the system
localises the evidence for its predictions, not just predicts. The predictive
harness (``evaluation/evaluate.py``) answers *what*; this one answers *where*.

It uses the NIH ChestX-ray14 ``BBox_List_2017.csv`` release (~984 hand-drawn
boxes over 8 of the 14 pathologies) as ground truth. For each boxed finding we:

1. run the classifier + explainer to produce a Grad-CAM / Score-CAM heatmap for
   *that* pathology,
2. rescale the ground-truth box from original-image pixels into the heatmap
   grid, and
3. score the heatmap against the box with two complementary metrics already
   defined in ``evaluation/metrics.py``:

   * **pointing game** — does the heatmap's peak fall inside the box?
   * **localization IoU** — overlap between the thresholded heatmap and the box,
     plus a localization-accuracy rate at an IoU threshold (the T(IoU) measure
     reported in the original ChestX-ray8 paper).

Results (per-pathology and overall) are written to ``evaluation/results/`` as
JSON so ``paper/`` can render them into a table.

Usage::

    python -m evaluation.evaluate_localization --config configs/default.yaml \
        --checkpoint models/checkpoints/densenet121_best.pt

The CSV path defaults to ``<data_root>/BBox_List_2017.csv``; override it with
``--bbox-csv``. Use ``--limit N`` for a quick smoke run over the first N boxes.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np  # hard dependency, also imported unguarded by metrics.py

try:
    import torch
    import pandas as pd
except ImportError:  # pragma: no cover - exercised only without the ML stack
    torch = None
    pd = None

from models.common.config import Config
from models.common.constants import CHESTXRAY14_LABELS
from models.common.preprocessing import load_image
from evaluation.metrics import pointing_game, localization_iou
from evaluation.repro import reproducibility_info


# The BBox release spells one label differently from the canonical taxonomy in
# ``constants.py`` ("Infiltrate" vs "Infiltration"); normalise it so the class
# index lookup against CHESTXRAY14_LABELS succeeds.
BBOX_LABEL_ALIASES: dict[str, str] = {"Infiltrate": "Infiltration"}

# The 8 pathologies that actually carry boxes in BBox_List_2017.csv. The other
# six (Consolidation, Edema, Emphysema, Fibrosis, Pleural_Thickening, Hernia)
# have no localisation ground truth and cannot be scored here.
BOXED_LABELS: tuple[str, ...] = (
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
)


def to_xyxy(x: float, y: float, w: float, h: float) -> tuple[float, float, float, float]:
    """Convert an ``(x, y, w, h)`` box to ``(x0, y0, x1, y1)`` corners."""
    return (x, y, x + w, y + h)


def scale_box(
    box_xyxy: tuple[float, float, float, float],
    orig_size: tuple[int, int],
    cam_size: tuple[int, int],
) -> tuple[int, int, int, int]:
    """Map a box from original-image pixels into heatmap-grid pixels.

    The inference transform resizes the whole image to ``image_size`` without
    cropping or preserving aspect ratio, so the mapping is a simple per-axis
    scale. ``orig_size`` and ``cam_size`` are ``(width, height)``. The returned
    corners are integer and clipped to the heatmap bounds; ``x1``/``y1`` are
    exclusive (suitable for array slicing).
    """
    ow, oh = orig_size
    cw, ch = cam_size
    sx = cw / ow if ow else 0.0
    sy = ch / oh if oh else 0.0
    x0, y0, x1, y1 = box_xyxy
    nx0 = max(0, min(int(round(x0 * sx)), cw - 1))
    ny0 = max(0, min(int(round(y0 * sy)), ch - 1))
    nx1 = max(nx0 + 1, min(int(round(x1 * sx)), cw))
    ny1 = max(ny0 + 1, min(int(round(y1 * sy)), ch))
    return (nx0, ny0, nx1, ny1)


def score_box(
    cam,
    gt_box_xyxy: tuple[int, int, int, int],
    cam_threshold: float = 0.5,
    iou_threshold: float = 0.1,
) -> dict:
    """Score one heatmap against one ground-truth box.

    Returns the pointing-game hit, the IoU, and whether the IoU clears
    ``iou_threshold`` (the localisation-accuracy criterion).
    """
    hit = pointing_game(cam, gt_box_xyxy)
    iou = localization_iou(cam, gt_box_xyxy, threshold=cam_threshold)
    return {
        "pointing_hit": bool(hit),
        "iou": float(iou),
        "loc_correct": bool(iou >= iou_threshold),
    }


def _aggregate(records: list[dict]) -> dict:
    """Collapse per-box scores into pointing accuracy, mean IoU, loc accuracy."""
    n = len(records)
    if n == 0:
        return {"n": 0, "pointing_game": 0.0, "mean_iou": 0.0, "loc_accuracy": 0.0}
    return {
        "n": n,
        "pointing_game": float(np.mean([r["pointing_hit"] for r in records])),
        "mean_iou": float(np.mean([r["iou"] for r in records])),
        "loc_accuracy": float(np.mean([r["loc_correct"] for r in records])),
    }


def _load_bbox_table(csv_path: Path) -> "pd.DataFrame":
    """Read BBox_List_2017.csv robustly by column position.

    The released header is awkwardly comma-split inside ``Bbox [x,y,w,h]`` and
    carries trailing empty columns, so we index by position rather than name:
    col 0 = image, 1 = label, 2..5 = x, y, w, h.
    """
    raw = pd.read_csv(csv_path)
    cols = list(raw.columns)
    table = pd.DataFrame(
        {
            "image": raw[cols[0]],
            "label": raw[cols[1]].astype(str).str.strip(),
            "x": pd.to_numeric(raw[cols[2]], errors="coerce"),
            "y": pd.to_numeric(raw[cols[3]], errors="coerce"),
            "w": pd.to_numeric(raw[cols[4]], errors="coerce"),
            "h": pd.to_numeric(raw[cols[5]], errors="coerce"),
        }
    )
    return table.dropna(subset=["x", "y", "w", "h"]).reset_index(drop=True)


def run(
    config: Config,
    checkpoint: str | None,
    bbox_csv: Path,
    cam_threshold: float,
    iou_threshold: float,
    seed: int,
    limit: int | None,
) -> dict:
    """Run the localisation evaluation and return the summary dict."""
    if torch is None:
        raise RuntimeError("PyTorch, numpy, and pandas are required to evaluate.")

    # Deterministic explanations: fixed seeds so re-runs reproduce.
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Imported lazily so the pure helpers above stay importable without torch.
    from models.classification.infer import Classifier
    from models.explainability.explainer import Explainer

    image_dir = Path(config.data.data_root) / "images"
    table = _load_bbox_table(bbox_csv)
    if limit is not None:
        table = table.iloc[:limit].reset_index(drop=True)

    classifier = Classifier(
        checkpoint_path=checkpoint or config.model.checkpoint_path,
        backbone=config.model.backbone,
        image_size=config.data.image_size,
    )
    explainer = Explainer(
        model=classifier.model,
        backbone=config.model.backbone,
        method=config.explain.method,
        target_layer=config.explain.target_layer,
        image_size=config.data.image_size,
        overlay_alpha=config.explain.overlay_alpha,
        colormap=config.explain.colormap,
    )
    cam_size = (config.data.image_size, config.data.image_size)

    per_label: dict[str, list[dict]] = defaultdict(list)
    skipped = 0
    for row in table.itertuples(index=False):
        label = BBOX_LABEL_ALIASES.get(row.label, row.label)
        if label not in CHESTXRAY14_LABELS:
            skipped += 1
            continue
        image_path = image_dir / row.image
        if not image_path.exists():
            skipped += 1
            continue

        class_idx = CHESTXRAY14_LABELS.index(label)
        explanation = explainer.explain(image_path, label, class_idx)

        orig_size = load_image(image_path).size  # (width, height)
        gt_box = scale_box(to_xyxy(row.x, row.y, row.w, row.h), orig_size, cam_size)
        per_label[label].append(
            score_box(explanation.heatmap, gt_box, cam_threshold, iou_threshold)
        )

    all_records = [r for recs in per_label.values() for r in recs]
    summary = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "backbone": config.model.backbone,
        "explain_method": config.explain.method,
        "checkpoint": checkpoint or config.model.checkpoint_path,
        "cam_threshold": cam_threshold,
        "iou_threshold": iou_threshold,
        "seed": seed,
        "reproducibility": reproducibility_info(seed),
        "n_boxes": len(all_records),
        "n_skipped": skipped,
        "overall": _aggregate(all_records),
        "per_label": {label: _aggregate(recs) for label, recs in sorted(per_label.items())},
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate MIRROR explanations against ground-truth boxes."
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Trained classifier checkpoint. Without one, explanations come from "
        "ImageNet-pretrained weights and the numbers are not meaningful.",
    )
    parser.add_argument(
        "--bbox-csv",
        default=None,
        help="Path to BBox_List_2017.csv (default: <data_root>/BBox_List_2017.csv).",
    )
    parser.add_argument(
        "--cam-threshold",
        type=float,
        default=0.5,
        help="Heatmap value above which a pixel counts as 'activated' for IoU.",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.1,
        help="IoU at/above which a localisation counts as correct (T(IoU)).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--limit", type=int, default=None, help="Score only the first N boxes."
    )
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    bbox_csv = (
        Path(args.bbox_csv)
        if args.bbox_csv
        else Path(config.data.data_root) / "BBox_List_2017.csv"
    )
    if not bbox_csv.exists():
        raise SystemExit(
            f"Ground-truth box file not found: {bbox_csv}\n"
            "Download NIH ChestX-ray14's BBox_List_2017.csv into your data_root "
            "(see datasets/README.md) or pass --bbox-csv."
        )
    if args.checkpoint is None and config.model.checkpoint_path is None:
        print(
            "[loc-eval] WARNING: no checkpoint — explaining ImageNet-pretrained "
            "weights. Results exercise the harness but are not meaningful."
        )

    summary = run(
        config=config,
        checkpoint=args.checkpoint,
        bbox_csv=bbox_csv,
        cam_threshold=args.cam_threshold,
        iou_threshold=args.iou_threshold,
        seed=args.seed,
        limit=args.limit,
    )

    out_dir = Path("evaluation/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"loc_{config.model.backbone}_{config.explain.method}.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    overall = summary["overall"]
    print(
        f"pointing game = {overall['pointing_game']:.4f}  |  "
        f"mean IoU = {overall['mean_iou']:.4f}  |  "
        f"loc acc@IoU>={args.iou_threshold} = {overall['loc_accuracy']:.4f}  "
        f"(n={overall['n']}, skipped={summary['n_skipped']})"
    )
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
