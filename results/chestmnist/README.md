# ChestMNIST results: a real small-scale benchmark

These are **real** MIRROR results on **ChestMNIST** (MedMNIST v2): a CC BY 4.0
dataset derived from NIH ChestX-ray14, with the **same 14 pathologies in the same
label order**, multi-label, official splits, just downsampled. MIRROR's taxonomy
applies unchanged, so this is a legitimate small stand-in for the full 45 GB NIH
release. The numbers are genuine (real radiographs, real labels); they are a
**reduced-resolution (64 px source, upsampled to 224 px for the backbone),
reduced-budget** result: 8,000 studies sampled from the pooled official train and
validation splits, 10% held out, giving 7,200 training and 800 validation images,
best of a four-epoch CPU run. This is not the full-resolution NIH ChestX-ray14
benchmark, which needs the full release and a GPU.

## Setup

| | |
| --- | --- |
| Data | ChestMNIST 64px, 8,000 train/val + 12,000 test (seed 42) |
| Model | DenseNet-121, ImageNet-pretrained, 14-way multi-label head |
| Train | AdamW, lr 3e-4, cosine, dropout 0.2, batch 32, CPU; best of a 4-epoch run (val macro AUROC peaked at epoch 2) |
| Eval | official-style held-out test split (12,000 images), bootstrap 95% CIs (B=1000) |

## Headline (test split, N = 12,000)

| Metric | Value |
| --- | --- |
| **Macro AUROC (95% CI)** | **0.729 [0.718, 0.738]** |
| Macro AUPRC | 0.135 |
| Operating point @0.5 (sens/spec/PPV/NPV) | 0.019 / 0.997 / 0.097 / 0.950 |
| Calibration (Brier / ECE) | 0.045 / 0.018 |
| Ablation invariance (max prob delta) | **0.0** (exact) |

AUROC is the standard, threshold-independent ChestMNIST headline metric. **0.729**
is **0.04 below** the published MedMNIST v2 DenseNet-121 baseline (AUROC ≈0.77,
trained on all 78k images for many epochs on GPU); that baseline used far more
compute, and we have not tested whether more compute would close the gap.

At the fixed 0.5 threshold the model is not a working detector: **11 of the 14
labels emit no positive prediction at all** (sensitivity exactly 0.0, specificity
exactly 1.0). Only Effusion (sens 0.143), Infiltration (0.110), and Pneumothorax
(0.012) ever fire, and macro F1 is 0.031. That is a consequence of the reduced
training budget and 64px resolution, not a deliberately conservative operating
point.

The calibration figures should be read against a baseline, not on their own: a
constant predictor that always returns each label's base rate scores macro Brier
**0.0472** on this test split, against the model's **0.0453**. Recompute with
`python paper/calibration_baseline.py`.

## Per-label AUROC: clinically sensible ordering

The ranking tracks how visible each finding is, which is the reassuring outcome:
the classically easier findings score highest and the subtle small ones lowest.

| Pathology | AUROC (95% CI) | | Pathology | AUROC (95% CI) |
| --- | --- | --- | --- | --- |
| Edema | 0.849 [0.828, 0.871] | | Hernia | 0.725 [0.615, 0.820] |
| Cardiomegaly | 0.830 [0.806, 0.851] | | Mass | 0.696 [0.673, 0.719] |
| Effusion | 0.827 [0.817, 0.838] | | Fibrosis | 0.674 [0.639, 0.705] |
| Consolidation | 0.754 [0.736, 0.773] | | Pneumonia | 0.672 [0.627, 0.716] |
| Emphysema | 0.744 [0.713, 0.774] | | Pleural Thickening | 0.666 [0.638, 0.692] |
| Pneumothorax | 0.738 [0.719, 0.757] | | Infiltration | 0.660 [0.648, 0.673] |
| Atelectasis | 0.729 [0.716, 0.742] | | Nodule | 0.643 [0.622, 0.662] |

## Files

| File | Harness |
| --- | --- |
| `eval_chestmnist_densenet121.json` | `evaluation/evaluate.py`: full predictive panel + bootstrap CIs |
| `ablation_chestmnist_densenet121.json` | `evaluation/ablation.py`: invariance (max prob delta = 0) + latency |

Each carries a top-level `_note` restating what this is (and is not).

## Reproduce

```bash
python -m datasets.scripts.prepare_chestmnist --out datasets/raw/chestmnist \
    --n-train-val 8000 --n-test 12000 --size 64 --seed 42
python -m models.classification.train  --config configs/chestmnist.yaml
python -m evaluation.evaluate          --config configs/chestmnist.yaml --checkpoint models/checkpoints/densenet121_best.pt
python -m evaluation.ablation          --config configs/chestmnist.yaml --checkpoint models/checkpoints/densenet121_best.pt \
    --prediction-results evaluation/results/eval_densenet121.json --images datasets/raw/chestmnist/images --n 24
```

For the full-resolution NIH ChestX-ray14 numbers, use
`datasets/scripts/download_chestxray14.py` (45 GB) and a GPU.
