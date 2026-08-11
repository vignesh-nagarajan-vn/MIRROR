"""No-skill baselines for the ChestMNIST panel reported in the paper.

Arithmetic on already-committed data, not a new experiment. Two aggregate metrics
in the paper look healthy read on their own and stop looking healthy once the
no-skill floor is written next to them. Both floors follow from the label
prevalences, which the evaluation JSON already records as ``support_pos`` and
``support_neg``:

    p = support_pos / (support_pos + support_neg)

* **Brier.** A predictor that ignores the image and always emits ``p`` scores
  exactly ``p * (1 - p)`` on that label (the variance of a Bernoulli(p)).
  Averaging over the 14 labels gives the macro Brier of the no-skill baseline.
* **AUPRC.** A uniformly random ranker has expected average precision equal to
  the prevalence ``p``. So AUPRC is only meaningful as a ratio to ``p``, and the
  paper reports that lift.

Also emits the exact per-label panel typeset in the paper, so the table cannot
drift from the JSON.

Run:
    python paper/calibration_baseline.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVAL_JSON = ROOT / "results" / "chestmnist" / "eval_chestmnist_densenet121.json"
SYNTH_JSON = (
    ROOT / "results" / "synthetic_validation" / "eval_synthetic_densenet121.json"
)


def main() -> None:
    data = json.loads(EVAL_JSON.read_text(encoding="utf-8"))
    per_label = data["operating_point"]["per_label"]
    auroc = data["per_label_auroc"]
    auprc = data["per_label_auprc"]
    cis = data["per_label_auroc_ci"]

    rows = []
    for label, m in per_label.items():
        pos, neg = m["support_pos"], m["support_neg"]
        prev = pos / (pos + neg)
        rows.append(
            {
                "label": label,
                "pos": pos,
                "prev": prev,
                "brier_floor": prev * (1.0 - prev),
                "auroc": auroc[label],
                "lo": cis[label]["lo"],
                "hi": cis[label]["hi"],
                "auprc": auprc[label],
                "lift": auprc[label] / prev,
                "sens": m["sensitivity"],
                "spec": m["specificity"],
                "ppv": m["ppv"],
            }
        )

    silent = [r for r in rows if r["sens"] == 0.0]
    firing = [r for r in rows if r["sens"] > 0.0]

    print("=" * 100)
    print("PER-LABEL PANEL  (paste-ready for the paper table, ChestMNIST test split)")
    print("=" * 100)
    hdr = f"{'label':<19} {'prev':>6} {'AUROC':>6} {'95% CI':>15} {'AUPRC':>6} {'lift':>5} {'sens':>6} {'spec':>6} {'PPV':>6}"
    print(hdr)
    for r in sorted(rows, key=lambda r: -r["auroc"]):
        ci = f"({r['lo']:.3f}-{r['hi']:.3f})"
        ppv = f"{r['ppv']:.3f}" if r["sens"] > 0 else "  --"
        print(
            f"{r['label']:<19} {r['prev']:>6.3f} {r['auroc']:>6.3f} {ci:>15} "
            f"{r['auprc']:>6.3f} {r['lift']:>5.1f} {r['sens']:>6.3f} "
            f"{r['spec']:>6.3f} {ppv:>6}"
        )

    brier_floor = sum(r["brier_floor"] for r in rows) / len(rows)
    brier_model = data["calibration"]["brier"]
    macro_lift = sum(r["lift"] for r in rows) / len(rows)

    print()
    print("=" * 100)
    print("HEADLINE NUMBERS")
    print("=" * 100)
    print(f"n_test                                   = {data['n_test']}")
    print(f"macro AUROC                              = {data['macro_auroc']:.4f}")
    print(f"macro AUPRC                              = {data['macro_auprc']:.4f}")
    print(f"macro F1                                 = {data['macro_f1']:.4f}")
    print(f"macro sensitivity / specificity          = "
          f"{data['operating_point']['macro']['sensitivity']:.4f} / "
          f"{data['operating_point']['macro']['specificity']:.4f}")
    print()
    print(f"labels with sensitivity exactly 0.0      = {len(silent)} of {len(rows)}")
    print(f"  silent : {', '.join(r['label'] for r in silent)}")
    print(f"  firing : {', '.join(r['label'] for r in firing)}")
    print(f"  prevalence range of the silent labels  = "
          f"{min(r['prev'] for r in silent):.4f} to {max(r['prev'] for r in silent):.4f}")
    print()
    print(f"macro Brier, constant-prevalence model    = {brier_floor:.6f}")
    print(f"macro Brier, MIRROR DenseNet-121          = {brier_model:.6f}")
    print(f"  absolute difference                    = {brier_floor - brier_model:.6f}")
    print(f"  model as a fraction of the floor       = {brier_model / brier_floor:.4f}")
    print(f"macro ECE                                = {data['calibration']['ece']:.6f}")
    print()
    print(f"mean AUPRC lift over prevalence          = {macro_lift:.2f}x")
    print(f"  best  : {max(rows, key=lambda r: r['lift'])['label']} "
          f"{max(r['lift'] for r in rows):.1f}x")
    print(f"  worst : {min(rows, key=lambda r: r['lift'])['label']} "
          f"{min(r['lift'] for r in rows):.1f}x")

    # Synthetic harness control: the two groups of seven.
    synth = json.loads(SYNTH_JSON.read_text(encoding="utf-8"))
    sa = {k: v for k, v in synth["per_label_auroc"].items() if k != "macro"}
    ranked = sorted(sa.items(), key=lambda kv: -kv[1])
    signal, nosignal = ranked[:7], ranked[7:]
    ms = sum(v for _, v in signal) / 7
    mn = sum(v for _, v in nosignal) / 7
    print()
    print("=" * 100)
    print("SYNTHETIC HARNESS CONTROL")
    print("=" * 100)
    print(f"n_test                    = {synth['n_test']}")
    print(f"signal-bearing (7) mean   = {ms:.4f}   [{', '.join(f'{k} {v:.3f}' for k, v in signal)}]")
    print(f"no-signal      (7) mean   = {mn:.4f}   [{', '.join(f'{k} {v:.3f}' for k, v in nosignal)}]")
    print(f"cross-check (ms+mn)/2     = {(ms + mn) / 2:.5f}  vs JSON macro "
          f"{synth['per_label_auroc']['macro']:.5f}")


if __name__ == "__main__":
    main()
