"""Evaluate a trained MIRROR classifier on the ChestX-ray14 test split.

Computes per-label and macro AUROC plus macro F1, and writes a JSON summary to
``evaluation/results/``.

Usage:
    python -m evaluation.evaluate --config configs/default.yaml \
        --checkpoint models/checkpoints/densenet121_best.pt
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

try:
    import torch
    from torch.utils.data import DataLoader
    import numpy as np
except ImportError:  # pragma: no cover
    torch = None

from models.common.config import Config
from models.common.modalities import resolve_modality
from models.classification.model import build_model, load_checkpoint
from models.classification.dataset import ChestXray14Dataset
from evaluation.metrics import (
    macro_auroc,
    macro_auprc,
    f1_at_threshold,
    operating_point_metrics,
    brier_score,
    expected_calibration_error,
    bootstrap_cis,
)
from evaluation.repro import reproducibility_info


def collect_predictions(model, loader, device):
    model.eval()
    scores, targets = [], []
    with torch.no_grad():
        for images, y in loader:
            p = torch.sigmoid(model(images.to(device))).cpu().numpy()
            scores.append(p)
            targets.append(y.numpy())
    return np.concatenate(targets), np.concatenate(scores)


def main() -> None:
    if torch is None:
        raise RuntimeError("PyTorch is required to evaluate.")
    parser = argparse.ArgumentParser(description="Evaluate the MIRROR classifier.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--modality",
        default="chest X-ray",
        help="chest X-ray | brain MRI | CT — selects the label taxonomy used to "
        "name per-label metrics.",
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=1000,
        help="Bootstrap resamples for confidence intervals (0 disables).",
    )
    parser.add_argument("--ci", type=float, default=0.95, help="CI level, e.g. 0.95.")
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    spec = resolve_modality(args.modality)
    labels = list(spec.labels)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = build_model(
        backbone=config.model.backbone,
        num_classes=len(labels),
        pretrained=False,
    )
    load_checkpoint(model, args.checkpoint, device=device)
    model.to(device)

    test_set = ChestXray14Dataset(
        config.data.data_root, split="test", image_size=config.data.image_size
    )
    loader = DataLoader(
        test_set, batch_size=config.data.batch_size, num_workers=config.data.num_workers
    )

    y_true, y_score = collect_predictions(model, loader, device)
    auroc = macro_auroc(y_true, y_score)
    auprc = macro_auprc(y_true, y_score)
    f1 = f1_at_threshold(y_true, y_score, args.threshold)
    operating = operating_point_metrics(y_true, y_score, args.threshold, labels)
    brier = brier_score(y_true, y_score)
    ece = expected_calibration_error(y_true, y_score)

    def _name(k: str) -> str:
        return labels[int(k)] if k.isdigit() and int(k) < len(labels) else k

    named_auroc = {_name(k): v for k, v in auroc.items()}
    named_auprc = {_name(k): v for k, v in auprc.items()}
    summary = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "modality": spec.display_name,
        "modality_key": spec.key,
        "backbone": config.model.backbone,
        "checkpoint": args.checkpoint,
        "n_test": len(test_set),
        "threshold": args.threshold,
        "macro_auroc": auroc["macro"],
        "macro_auprc": auprc["macro"],
        "macro_f1": f1,
        "operating_point": operating,
        "calibration": {"brier": brier, "ece": ece},
        "per_label_auroc": named_auroc,
        "per_label_auprc": named_auprc,
        "reproducibility": reproducibility_info(args.seed),
    }

    # Bootstrap confidence intervals over the test set, so the headline numbers
    # carry uncertainty rather than being bare point estimates.
    if args.bootstrap > 0:
        cis = bootstrap_cis(
            y_true,
            y_score,
            threshold=args.threshold,
            n_boot=args.bootstrap,
            ci=args.ci,
            seed=args.seed,
        )
        summary["macro_auroc_ci"] = cis["macro_auroc"]
        summary["macro_f1_ci"] = cis["macro_f1"]
        summary["macro_sensitivity_ci"] = cis["macro_sensitivity"]
        summary["macro_specificity_ci"] = cis["macro_specificity"]
        summary["brier_ci"] = cis["brier"]
        summary["per_label_auroc_ci"] = {
            _name(k): v for k, v in cis["per_label_auroc"].items()
        }
        summary["bootstrap"] = {"n_boot": cis["n_boot"], "ci_level": cis["ci_level"]}

    out_dir = Path("evaluation/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    # Include the modality so brain-MRI / head-CT runs don't overwrite the chest
    # X-ray results (the default modality keeps the original filename).
    suffix = "" if spec.key == "chest_xray" else f"_{spec.key}"
    out_path = out_dir / f"eval_{config.model.backbone}{suffix}.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    op = summary["operating_point"]["macro"]
    if args.bootstrap > 0:
        a, fci = summary["macro_auroc_ci"], summary["macro_f1_ci"]
        pct = int(args.ci * 100)
        print(
            f"macro AUROC = {auroc['macro']:.4f} "
            f"[{a['lo']:.4f}, {a['hi']:.4f}] {pct}% CI  |  "
            f"macro F1 = {f1:.4f} [{fci['lo']:.4f}, {fci['hi']:.4f}]"
        )
    else:
        print(f"macro AUROC = {auroc['macro']:.4f}  |  macro F1 = {f1:.4f}")
    print(
        f"macro AUPRC = {auprc['macro']:.4f}  |  "
        f"sensitivity = {op['sensitivity']:.4f}  |  "
        f"specificity = {op['specificity']:.4f}  |  "
        f"PPV = {op['ppv']:.4f}  |  NPV = {op['npv']:.4f}"
    )
    print(f"Brier = {brier:.4f}  |  ECE = {ece:.4f}  (calibration; lower is better)")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
