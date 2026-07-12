# Synthetic end-to-end validation run

> [!CAUTION]
> **These numbers are NOT a ChestX-ray14 benchmark and must never be cited as
> MIRROR's clinical performance.** They come from training and evaluating the real
> pipeline on a **synthetic** dataset of procedurally generated chest-X-ray-*like*
> images. They characterise the *pipeline* — that it trains, evaluates, localizes,
> and produces well-formed metrics end to end — not diagnostic accuracy on real
> radiographs. The paper's ChestX-ray14 result tables remain intentionally
> unfilled (red) pending a real trained checkpoint.

## What this is

The rest of the repository ships *illustrative* result snapshots (hand-written,
literature-scale, format-only). This folder is different: it is the **actual output
of the real harnesses run on real (if synthetic) data**, produced by executing the
full pipeline on this machine (CPU). It exists to prove the experimental machinery
works and to show *what the harnesses produce*, with genuine reproducibility stamps.

Pipeline exercised end to end:

```
generate 1400 synthetic images  ->  train DenseNet121 (6 epochs, CPU)
   ->  evaluate on held-out synthetic test split (350 images)
   ->  ablation (invariance + latency)
```

Exact commands are in [Reproducing](#reproducing) below; all settings are pinned in
[`configs/synthetic.yaml`](../../configs/synthetic.yaml). The dataset and checkpoint
are git-ignored (regenerable from seed 42); only these result JSONs are committed.

## Headline numbers (synthetic test split, N = 350)

| Metric | Value |
| --- | --- |
| Best validation macro AUROC (training) | **0.749** |
| Test macro AUROC (bootstrap 95% CI) | **0.725 [0.701, 0.749]** |
| Test macro AUPRC | 0.233 |
| Operating point @ 0.5 (sens / spec / PPV / NPV) | 0.099 / 0.994 / 0.061 / 0.945 |
| Calibration (Brier / ECE) | 0.051 / 0.011 |
| Ablation invariance (max prob delta) | **0.0** (exact) |

The model is deliberately conservative at the 0.5 threshold (high specificity, low
sensitivity), so threshold-dependent F1/sensitivity are low even though ranking
(AUROC) is strong — a real, interpretable property, not a bug.

## The key validation: the model learns *exactly* the injected signal

The synthetic generator injects a visible blob for a subset of labels
(Mass/Nodule = small blob, Effusion/Infiltration/Consolidation/Edema = large blob,
Cardiomegaly = medium blob) and leaves the other seven labels with **no visual
correlate**. A correct pipeline should therefore learn the first group and sit at
chance for the second — which is exactly what happens:

| Group | Mean AUROC | Per-label |
| --- | --- | --- |
| **Has injected visual signal** | **0.917** | Mass 0.979, Nodule 0.974, Cardiomegaly 0.908, Consolidation 0.903, Effusion 0.895, Infiltration 0.894, Edema 0.866 |
| **No visual signal (expect ≈0.5)** | **0.557** | Hernia 0.620, Fibrosis 0.595, Emphysema 0.550, Pneumonia 0.540, Atelectasis 0.524, Pneumothorax 0.467, Pleural_Thickening 0.434 |

The clean split (0.917 vs 0.557) is the point: it shows the classifier, the loss,
the metrics, and the bootstrap CIs all behave correctly and that nothing is leaking
or fabricated. On real ChestX-ray14 the same harness would produce the same shape of
output over real anatomy.

## Files

| File | Harness | Notes |
| --- | --- | --- |
| `eval_synthetic_densenet121.json` | `evaluation/evaluate.py` | full predictive panel + bootstrap CIs |
| `ablation_synthetic_densenet121.json` | `evaluation/ablation.py` | invariance (max prob delta = 0) + live latency |

Each carries a top-level `_note` restating that it is a synthetic validation, not a
benchmark.

## Reproducing

**This synthetic run** (no downloads, CPU, ~10 min):

```bash
python -m datasets.scripts.make_synthetic_samples --out datasets/raw/synthetic_chestxray14 -n 1400 --seed 42
python -m models.classification.train    --config configs/synthetic.yaml
python -m evaluation.evaluate            --config configs/synthetic.yaml --checkpoint models/checkpoints/densenet121_best.pt
python -m evaluation.ablation            --config configs/synthetic.yaml --checkpoint models/checkpoints/densenet121_best.pt \
    --prediction-results evaluation/results/eval_densenet121.json --images datasets/raw/synthetic_chestxray14/images --n 24
```

**Real ChestX-ray14 numbers** (needs the ~45 GB licensed NIH release and a GPU):

```bash
python -m datasets.scripts.download_chestxray14 --data-root datasets/raw/chestxray14   # NIH data-use terms apply
# point configs/default.yaml data.data_root -> datasets/raw/chestxray14, train.device: cuda
python -m models.classification.train --config configs/default.yaml
python -m evaluation.evaluate              --config configs/default.yaml --checkpoint models/checkpoints/densenet121_best.pt
python -m evaluation.evaluate_localization --config configs/default.yaml --checkpoint models/checkpoints/densenet121_best.pt   # needs BBox_List_2017.csv
python -m evaluation.ablation              --config configs/default.yaml \
    --prediction-results evaluation/results/eval_densenet121.json \
    --localization-results evaluation/results/loc_densenet121_gradcam.json
python -m evaluation.aggregate_seeds evaluation/results/eval_densenet121_seed*.json
```

Then copy the resulting numbers into the `paper/main.tex` tables and set them black.
```
