# Evaluation

MIRROR's research question has two halves — does adding explanation/reasoning
layers help *interpretability* without hurting *prediction*? — so evaluation
comes in two harnesses that write JSON summaries to `evaluation/results/`
(git-ignored) for the `paper/` to tabulate.

## 1. Predictive quality — `evaluate.py`

Per-label and macro **AUROC** plus macro **F1** on the official ChestX-ray14
test split, each with a **bootstrap 95% confidence interval** over the test set.

```bash
python -m evaluation.evaluate --config configs/default.yaml \
    --checkpoint models/checkpoints/densenet121_best.pt
# control the bootstrap (0 disables) and CI level:
python -m evaluation.evaluate --config configs/default.yaml \
    --checkpoint models/checkpoints/densenet121_best.pt --bootstrap 2000 --ci 0.95
```

Writes `results/eval_<backbone>.json`, including `macro_auroc_ci`, `macro_f1_ci`,
`per_label_auroc_ci` (each `{mean, std, lo, hi}`), and a `reproducibility` block
(seed, git commit, library versions) so every number can be regenerated. The run
is seeded with `--seed` (default 42).

The bootstrap (`bootstrap_cis` in `metrics.py`) resamples the N test examples
with replacement; one resample drives all metrics, so the intervals are mutually
consistent. Point estimates alone can't support "config A beats config B" — the
CIs are what make the comparison defensible.

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

## 4. Multi-seed aggregation — `aggregate_seeds.py`

Bootstrap CIs capture test-set sampling noise for *one* model; they say nothing
about training stochasticity. For that, train under several seeds and summarise
across them as mean ± std — what reviewers expect on ChestX-ray14.

```bash
for s in 0 1 2; do
    python -m models.classification.train --config configs/default.yaml --seed $s
    python -m evaluation.evaluate --config configs/default.yaml \
        --checkpoint models/checkpoints/densenet121_best.pt
    cp evaluation/results/eval_densenet121.json evaluation/results/eval_seed$s.json
done
python -m evaluation.aggregate_seeds evaluation/results/eval_seed*.json
```

`train.py` now takes `--seed` to override `train.seed` (and records the seed in
the checkpoint). `aggregate_seeds.py` reads each per-seed `eval_*.json` and writes
`results/aggregate_<backbone>.json` with mean ± std for macro AUROC/F1 and every
per-label AUROC. Report the two uncertainties together: bootstrap CI (per model)
and seed std (across models).

## Metric definitions

All metrics live in [`metrics.py`](metrics.py): `macro_auroc`, `f1_at_threshold`,
`bootstrap_cis`, `pointing_game`, and `localization_iou`. The localization harness
is a thin driver over the localization metrics plus box loading; the ablation
harness reuses the JSON the others emit; `aggregate_seeds.py` summarises across
seeds. Provenance for every run comes from [`repro.py`](repro.py).

## Tests

- `tests/test_localization_eval.py` — box scaling, per-box scoring, aggregation.
- `tests/test_ablation.py` — conditions, capability matrix, result merging, table
  assembly.
- `tests/test_metrics_bootstrap.py` — CI structure, determinism, ordering.
- `tests/test_aggregate_seeds.py` — multi-seed mean/std math.
- `tests/test_repro.py` — provenance record fields.

All cover the torch-free logic with synthetic inputs, so they run without a model
or any dataset.
