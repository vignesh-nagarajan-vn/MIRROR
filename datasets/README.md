# Datasets

MIRROR is developed against one **primary** dataset for benchmarking and several
**secondary** datasets for breadth across modalities. None of these are
redistributed in this repository — each must be obtained from its original source
under its own license. Download into `datasets/raw/<name>/` and keep processed
artifacts in `datasets/processed/`.

## Run it now — synthetic samples

The real datasets are large and license-restricted, so the repo ships a tiny
**synthetic** stand-in under `datasets/samples/chestxray14/` that mirrors the NIH
layout exactly (images + `Data_Entry_2017.csv` + split lists). It lets the demo,
the dataset loader, a training smoke test, and the DICOM ingest path all run with
zero downloads. Regenerate it any time:

```bash
python -m datasets.scripts.make_synthetic_samples --out datasets/samples/chestxray14 -n 24
```

> These images are fabricated noise shaped like a thorax — **not** radiographs and
> with no diagnostic meaning. They exist solely to exercise the plumbing. One of
> them (`synth_0000.dcm`) is a MONOCHROME1 **DICOM** so the `.dcm` ingest path is
> covered. Point `data.data_root` at `datasets/samples/chestxray14` in your config
> to train/evaluate against them.

## DICOM ingest

MIRROR reads native **DICOM** (`.dcm`) directly — `models/common/dicom.py` applies
the modality LUT (rescale slope/intercept), the VOI LUT (window center/width),
and MONOCHROME1 inversion, then hands a display-ready RGB image to the same
pipeline used for PNG/JPEG. Compressed transfer syntaxes (JPEG/JPEG2000) need an
extra handler — `pip install pylibjpeg pylibjpeg-libjpeg` (or `python-gdcm`);
uncompressed DICOM needs only pydicom.

## Primary — NIH ChestX-ray14

- **Scale:** 112,120 frontal chest X-rays from 30,805 patients.
- **Labels:** 14 disease categories (multi-label), plus "No Finding".
- **Why primary:** large, standardized, widely benchmarked — ideal for initial
  model development and reproducible comparison.
- **Source:** NIH Clinical Center release ("ChestX-ray8/14").

Fetch the real release (12 archives, ~45 GB) into the expected layout:

```bash
python -m datasets.scripts.download_chestxray14 --data-root datasets/raw/chestxray14
# or grab just the first 2 archives while developing:
python -m datasets.scripts.download_chestxray14 --data-root datasets/raw/chestxray14 --max-archives 2
```

Expected layout (what `models/classification/dataset.py` reads):

```
datasets/raw/chestxray14/
├── images/                 # all *.png, flat
├── Data_Entry_2017.csv     # official per-image metadata + Finding Labels
├── BBox_List_2017.csv      # official lesion boxes (for localization eval)
├── train_val_list.txt      # official train/val image IDs
└── test_list.txt           # official held-out test image IDs
```

The 14 labels, in canonical order, live in `models/common/constants.py`:
Atelectasis, Cardiomegaly, Effusion, Infiltration, Mass, Nodule, Pneumonia,
Pneumothorax, Consolidation, Edema, Emphysema, Fibrosis, Pleural_Thickening,
Hernia.

### Localization ground truth

The NIH release ships `BBox_List_2017.csv` — **~984 hand-drawn bounding boxes**
over **8** of the 14 pathologies (Atelectasis, Cardiomegaly, Effusion,
Infiltration, Mass, Nodule, Pneumonia, Pneumothorax; the other six have no box
ground truth). This is the ground truth `evaluation/evaluate_localization.py`
scores the Grad-CAM/Score-CAM heatmaps against (pointing game, IoU). It is part
of the standard ChestX-ray14 download; place it in `data_root` alongside
`Data_Entry_2017.csv`. Boxes are in original-image pixel coordinates
(1024×1024); the harness rescales them into the model's heatmap grid. Note the
CSV labels the infiltration class "Infiltrate" — the harness normalizes this to
the canonical "Infiltration". See [`evaluation/README.md`](../evaluation/README.md).

Verify a local copy and build a tiny smoke-test split:

```bash
python -m datasets.scripts.prepare_chestxray14 --data-root datasets/raw/chestxray14 --verify
python -m datasets.scripts.prepare_chestxray14 --data-root datasets/raw/chestxray14 --make-sample 200
```

## Secondary datasets

| Dataset | Modality | Role in MIRROR |
| --- | --- | --- |
| **RSNA Pneumonia Detection Challenge** | Chest X-ray | Pneumonia localization; cross-dataset generalization check |
| **MIMIC-CXR** | Chest X-ray + free-text reports | Report-generation supervision and evaluation against real radiologist reports |
| **Brain Tumor MRI Dataset** | Brain MRI | Extends the pipeline beyond chest film to a second modality |
| **COVID-19 Radiography Database** | Chest X-ray | Additional pathology class and robustness testing |

Each secondary dataset has its own label schema. To add one, implement a
`Dataset` subclass alongside `ChestXray14Dataset` and register its label list in
`models/common/constants.py`. The classifier head's `num_classes` and the report
generator's label descriptions should be updated together.

## Licensing & ethics

These datasets contain medical images governed by their respective data use
agreements. Do not commit raw images or PHI to version control — `datasets/raw/`
and `datasets/processed/` are git-ignored by default. MIRROR is a research
prototype and is **not** a medical device.
