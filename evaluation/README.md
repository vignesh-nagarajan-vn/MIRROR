# Evaluation

MIRROR's research question has two halves — does adding explanation/reasoning
layers help *interpretability* without hurting *prediction*? — so evaluation
comes in two harnesses that write JSON summaries to `evaluation/results/`
(git-ignored) for the `paper/` to tabulate.

## 1. Predictive quality — `evaluate.py`

Per-label and macro **AUROC** plus macro **F1** on the official ChestX-ray14
test split.

```bash
python -m evaluation.evaluate --config configs/default.yaml \
    --checkpoint models/checkpoints/densenet121_best.pt
```

Writes `results/eval_<backbone>.json`.

## 2. Explanation quality — `evaluate_localization.py`

This is the experiment behind MIRROR's central novelty claim: the system points
at the evidence for its predictions, not just predicts. It scores each
Grad-CAM / Score-CAM heatmap against NIH's hand-drawn boxes.

```bash
python -m evaluation.evaluate_localization --config configs/default.yaml \
    --checkpoint models/checkpoints/densenet121_best.pt
# quick smoke run over the first 20 boxes:
python -m evaluation.evaluate_localization --config configs/default.yaml \
    --checkpoint models/checkpoints/densenet121_best.pt --limit 20
```

Writes `results/loc_<backbone>_<method>.json` with per-pathology and overall:

| Metric | Question it answers | Definition |
| --- | --- | --- |
| **pointing game** | Does the heatmap point at the lesion at all? | Fraction of boxes whose region contains the heatmap's peak pixel. |
| **mean IoU** | How well does the highlighted region match the lesion? | Mean intersection-over-union between the box and the heatmap thresholded at `--cam-threshold` (default 0.5). |
| **localization accuracy** | How often is the overlap "good enough"? | Fraction of boxes with IoU ≥ `--iou-threshold` (default 0.1 — the T(IoU) measure from the ChestX-ray8 paper). |

### Ground truth

Localization uses NIH's `BBox_List_2017.csv` (~984 boxes over **8** of the 14
pathologies: Atelectasis, Cardiomegaly, Effusion, Infiltration, Mass, Nodule,
Pneumonia, Pneumothorax). The other six labels have no box ground truth and are
not scored. The CSV defaults to `<data_root>/BBox_List_2017.csv`; override with
`--bbox-csv`. See [`datasets/README.md`](../datasets/README.md#localization-ground-truth).

### Notes for reproducibility

- Runs are seeded (`--seed`, default 42) so re-runs reproduce.
- Without a trained `--checkpoint`, the harness explains ImageNet-pretrained
  weights: it exercises the full path but the numbers are not meaningful.
- Boxes are scaled from original-image pixels into the heatmap grid assuming the
  inference transform's plain resize (no crop / aspect change) — matching
  `models/common/preprocessing.py`.

## 3. Ablation — `ablation.py`

The research question is comparative — does adding the localization and reasoning
layers help *without* degrading prediction? — so it needs the baseline it names.
`ablation.py` defines three conditions and assembles the side-by-side table:

| Condition | Prediction | Localization | Report |
| --- | :---: | :---: | :---: |
| `classification_only` | ✓ | — | — |
| `with_localization` | ✓ | ✓ | — |
| `full_mirror` | ✓ | ✓ | ✓ |

```bash
python -m evaluation.ablation --config configs/default.yaml \
    --prediction-results evaluation/results/eval_densenet121.json \
    --localization-results evaluation/results/loc_densenet121_gradcam.json
# assemble the table without running a model (capabilities + supplied metrics only):
python -m evaluation.ablation --config configs/default.yaml --no-latency \
    --prediction-results evaluation/results/eval_densenet121.json
```

Writes `results/ablation_<backbone>.json`. What the table demonstrates:

- **No predictive cost.** Layers 2–3 are post-hoc, so AUROC/F1 are identical in
  every row. The harness verifies this empirically by running all three
  conditions on a sample and checking the predictions are unchanged
  (`predictions_invariant`, `max_prob_delta`).
- **Added capability + its latency.** Localization metrics populate only the rows
  whose localization layer is on; the report column reflects the reasoning layer;
  and a live profile records per-stage wall-clock latency per condition.

Predictive and localization numbers are read from the JSON written by harnesses 1
and 2 (nothing is recomputed). The capability matrix and latency profile run on
the bundled synthetic samples (`--images`, default
`datasets/samples/chestxray14/images`), so the table is partially reproducible
with no downloads. The conditions map directly onto the new `localize` / `report`
flags on `MirrorPipeline.analyze()`.

## Metric definitions

All metrics live in [`metrics.py`](metrics.py): `macro_auroc`, `f1_at_threshold`,
`pointing_game`, and `localization_iou`. The localization harness is a thin
driver over the latter two, plus box loading and aggregation; the ablation harness
reuses the JSON they emit.

## Tests

- `tests/test_localization_eval.py` — box scaling, per-box scoring, aggregation.
- `tests/test_ablation.py` — conditions, capability matrix, result merging, table
  assembly.

Both cover the torch-free logic with synthetic inputs, so they run without a model
or any dataset.
