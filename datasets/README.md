# Datasets

MIRROR is developed against one **primary** dataset for benchmarking and several
**secondary** datasets for breadth across modalities. None of these are
redistributed in this repository — each must be obtained from its original source
under its own license. Download into `datasets/raw/<name>/` and keep processed
artifacts in `datasets/processed/`.

## Primary — NIH ChestX-ray14

- **Scale:** 112,120 frontal chest X-rays from 30,805 patients.
- **Labels:** 14 disease categories (multi-label), plus "No Finding".
- **Why primary:** large, standardized, widely benchmarked — ideal for initial
  model development and reproducible comparison.
- **Source:** NIH Clinical Center release ("ChestX-ray8/14").

Expected layout (what `models/classification/dataset.py` reads):

```
datasets/raw/chestxray14/
├── images/                 # all *.png, flat
├── Data_Entry_2017.csv     # official per-image metadata + Finding Labels
├── train_val_list.txt      # official train/val image IDs
└── test_list.txt           # official held-out test image IDs
```

The 14 labels, in canonical order, live in `models/common/constants.py`:
Atelectasis, Cardiomegaly, Effusion, Infiltration, Mass, Nodule, Pneumonia,
Pneumothorax, Consolidation, Edema, Emphysema, Fibrosis, Pleural_Thickening,
Hernia.

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
