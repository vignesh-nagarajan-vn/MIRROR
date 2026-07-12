<div align="center">

# MIRROR

**Multimodal Intelligent Radiology Reasoning and Observation Reporter**

*An explainable medical-AI system that reads radiological images, localizes its
own evidence, and writes a clinician-style draft report. Completed as part of the
**Global Indian Scientists & Technocrats (GIST) 2026 Summer Internship Program.***

Mentored by **Mr. Sriram Venkatapathy** (AI Research at Capital One, PhD-CS at IIT Hyderabad)

**Live Demo:** https://mirror-ten-jet.vercel.app/

</div>

---

## Tech Stack

<!--- ML / AI --->
<p align="center"><sub><b>AI / ML & Frameworks:</b></sub></p>
<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white">
  <img src="https://img.shields.io/badge/torchvision-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white">
  <img src="https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white">
  <img src="https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white">
  <img src="https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white">
  <img src="https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black">
  <img src="https://img.shields.io/badge/Pillow-%23306998.svg?style=for-the-badge&logo=python&logoColor=white">
  <img src="https://img.shields.io/badge/pydicom%20%2F%20DICOM-%2300A4E4.svg?style=for-the-badge&logo=dicom&logoColor=white">
  <img src="https://img.shields.io/badge/Jupyter-F37626.svg?style=for-the-badge&logo=Jupyter&logoColor=white">
</p>

<!--- Backend --->
<p align="center"><sub><b>Backend:</b></sub></p>
<p align="center">
  <img src="https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54">
  <img src="https://img.shields.io/badge/FastAPI-005571.svg?style=for-the-badge&logo=fastapi&logoColor=white">
  <img src="https://img.shields.io/badge/Uvicorn-%23499848.svg?style=for-the-badge&logo=gunicorn&logoColor=white">
  <img src="https://img.shields.io/badge/Pydantic-E92063?style=for-the-badge&logo=pydantic&logoColor=white">
</p>

<!--- Frontend --->
<p align="center"><sub><b>Frontend:</b></sub></p>
<p align="center">
  <img src="https://img.shields.io/badge/Next.js-black?style=for-the-badge&logo=next.js&logoColor=white">
  <img src="https://img.shields.io/badge/react-%2320232a.svg?style=for-the-badge&logo=react&logoColor=%2361DAFB">
  <img src="https://img.shields.io/badge/typescript-%23007ACC.svg?style=for-the-badge&logo=typescript&logoColor=white">
  <img src="https://img.shields.io/badge/node.js-6DA55F?style=for-the-badge&logo=node.js&logoColor=white">
  <img src="https://img.shields.io/badge/npm-%23CB3837.svg?style=for-the-badge&logo=npm&logoColor=white">
</p>

<!--- Data / Tooling --->
<p align="center"><sub><b>Data & Tooling:</b></sub></p>
<p align="center">
  <img src="https://img.shields.io/badge/postgresql-%234169E1.svg?style=for-the-badge&logo=postgresql&logoColor=white">
  <img src="https://img.shields.io/badge/Supabase-3ECF8E?style=for-the-badge&logo=supabase&logoColor=white">
  <img src="https://img.shields.io/badge/yaml-%23ffffff.svg?style=for-the-badge&logo=yaml&logoColor=151515">
  <img src="https://img.shields.io/badge/pytest-%230A9EDC.svg?style=for-the-badge&logo=pytest&logoColor=white">
  <img src="https://img.shields.io/badge/git-%23F05033.svg?style=for-the-badge&logo=git&logoColor=white">
</p>

<!--- Paper / Writing --->
<p align="center"><sub><b>Paper & Writing:</b></sub></p>
<p align="center">
  <img src="https://img.shields.io/badge/LaTeX-008080.svg?style=for-the-badge&logo=latex&logoColor=white">
  <img src="https://img.shields.io/badge/Overleaf-47A141.svg?style=for-the-badge&logo=overleaf&logoColor=white">
  <img src="https://img.shields.io/badge/BibTeX-008080.svg?style=for-the-badge&logo=latex&logoColor=white">
  <img src="https://img.shields.io/badge/arXiv-B31B1B.svg?style=for-the-badge&logo=arxiv&logoColor=white">
</p>

## What MIRROR does

Most medical-imaging models stop at a prediction. MIRROR adds the two layers that
make a prediction *trustworthy and usable*: it shows **where** the evidence is and
explains **what it means** in plain clinical language.

It analyzes a radiological study across **three modalities** (chest X-ray, brain
MRI, and head CT), identifies potential abnormalities from that modality's finding
taxonomy, highlights the diagnostic evidence with saliency overlays, and generates
a structured natural-language report in the right clinical vocabulary.

```
Image → Prediction → Evidence Localization → Clinical Reasoning → Human-Readable Report
```

## Research question

> Can multimodal AI systems that combine image classification, visual
> explainability, and language generation improve interpretability and user trust
> in medical image analysis compared to classification-only approaches?

The repo is built to *measure* this: predictive metrics (AUROC/F1) and
explanation metrics (pointing game / localization IoU) live side by side in
`evaluation/`.

## Why it's different (novelty)

While most medical-imaging systems focus solely on disease classification, MIRROR
integrates three complementary layers into one framework:

1. **Radiological image understanding**: a CNN/ViT classifier.
2. **Visual evidence localization**: Grad-CAM / Score-CAM saliency.
3. **Natural-language clinical report generation**: an LLM (or offline template)
   that reasons *only* over the structured evidence above.

The result not only predicts abnormalities but communicates **why** the
prediction was made and **how** it relates to potential clinical findings; every
finding it reports traces back to a probability and a saliency region (the
descriptive prose around each finding is model-generated and not pixel-verified).

## System architecture

MIRROR turns a single radiograph into a reviewable diagnostic draft by chaining
**three complementary layers**, where each layer's output becomes the *grounded
input* to the next, so the final report can always be traced back to a
probability and a specific image region.

![Architecture](docs/images/architecture.svg)

| Layer | Module | Does | Produces |
| --- | --- | --- | --- |
| **1 · Classification** | [`models/classification/`](models/classification/) | A CNN/ViT backbone (DenseNet121 · EfficientNet-B0 · ViT-B/16) with a multi-label head sized to the study's modality | Per-label probabilities |
| **2 · Evidence localization** | [`models/explainability/`](models/explainability/) | Grad-CAM / Score-CAM hooks the target layer for each positive label | Heatmap + region (centroid, bbox) |
| **3 · Clinical reasoning** | [`models/report_generation/`](models/report_generation/) | An LLM (or an offline template) prompts over the **structured evidence only, never the pixels** | `FINDINGS` / `IMPRESSION` report |

The pipeline is modality-agnostic; a single registry
([`models/common/modalities.py`](models/common/modalities.py)) supplies the finding
taxonomy, the anatomical vocabulary, and the report phrasing for each modality:

| Modality | Findings | Grounded in | Location vocabulary |
| --- | --- | --- | --- |
| **Chest X-ray** | 14 | NIH ChestX-ray14 | lung zones (frontal) |
| **Brain MRI** | 11 | Brain Tumor MRI Dataset + routine neuro findings | lobar regions (axial) |
| **Head CT** | 11 | RSNA Intracranial Hemorrhage + routine acute-CT findings | lobar regions (axial) |

Modality is chosen in the UI (or via `--modality`), or auto-detected from a DICOM
`Modality` tag with `--modality auto` (`MR`→brain MRI, `CT`→head CT, `CR`/`DX`→chest).

[`models/pipeline.py`](models/pipeline.py) orchestrates the four stages into one
`AnalysisResult`. The two later layers are individually toggleable, which is
exactly what recovers the ablation conditions the research question names
(classification-only → +localization → full MIRROR) and lets the evaluation
harnesses show *added interpretability at no predictive cost*.

**Two interchangeable serving engines satisfy the same response contract**, so
the UI is identical in both:

- **Local full stack:** FastAPI ([`backend/`](backend/)) wraps the real PyTorch
  pipeline; the frontend points at it via `NEXT_PUBLIC_API_URL`. Every input type
  (PNG/JPEG/BMP/WEBP + DICOM) and rendered Grad-CAM overlays.
- **Hosted on Vercel:** a Next.js serverless route
  ([`frontend/app/api/analyze/route.ts`](frontend/app/api/analyze/route.ts)) uses
  **Claude's vision model** as a drop-in engine (the PyTorch pipeline can't fit
  serverless), returning the same JSON with a bounding box per finding.

For the full write-up (per-layer module breakdowns, the grounding rationale, and
the deployment topology table), see [`docs/architecture.md`](docs/architecture.md).

## Live demo (v1.1.0)

The hosted build at **[mirror-ten-jet.vercel.app](https://mirror-ten-jet.vercel.app/)**
runs the full pipeline in the browser via a Next.js serverless route backed by
Claude vision (`claude-haiku-4-5`): upload a study, score the modality's finding
taxonomy, draw a bounding box per positive finding, and draft a grounded
`FINDINGS` / `IMPRESSION` report.

![Study input and predictions](docs/images/deployment/mirror-sample1-input-predictions-ui.png)

A full annotated walkthrough (evidence overlays and the draft report on a real
session) is in [`docs/deployment-showcase.md`](docs/deployment-showcase.md).

## Quickstart

Run the full pipeline on a bundled sample with zero downloads (ImageNet weights,
offline report backend, works anywhere):

```bash
git clone https://github.com/vignesh-nagarajan-vn/MIRROR.git && cd MIRROR
python -m venv .venv && source .venv/Scripts/activate   # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python -m demo.run_demo datasets/samples/chestxray14/images/synth_0001.png
# other modalities (or --modality auto to route a DICOM by its Modality tag):
python -m demo.run_demo datasets/samples/brain_mri/images/mri_0001.png --modality "brain MRI"
python -m demo.run_demo datasets/samples/head_ct/images/ct_0000.dcm --modality auto
```

This prints predictions and a draft report and writes Grad-CAM overlays to
`demo/assets/`. Inputs may be PNG/JPEG/BMP/WEBP or DICOM (`.dcm`); on ImageNet
weights the predictions are structurally valid but not diagnostic.

- **Full local stack** (FastAPI backend + Next.js reading-room UI):
  [`docs/setup.md`](docs/setup.md).
- **Deploy a live public site** on Vercel, powered by Claude vision with no backend
  to host: [`docs/deployment.md`](docs/deployment.md).
- **Richer prose reports** (optional): set `ANTHROPIC_API_KEY` and
  `report.provider: anthropic` in [`configs/default.yaml`](configs/default.yaml); it
  falls back to the offline template if the key is missing.

## Repository layout

```
mirror/
├── frontend/                     # Next.js "reading-room" UI
│   ├── app/
│   │   ├── page.tsx              # the single-page reading room
│   │   ├── layout.tsx           # root layout + fonts
│   │   └── api/analyze/route.ts  # serverless analyze route (Vercel, Claude vision)
│   ├── components/               # UploadPanel · FilmViewer · FindingsList · ReportPanel
│   ├── lib/api.ts                # typed client + response contract
│   ├── styles/globals.css        # reading-room theme
│   ├── vercel.json               # Vercel build/function config
│   └── .env.local.example        # NEXT_PUBLIC_API_URL + ANTHROPIC_API_KEY
├── backend/                      # FastAPI service (lazy-loads the pipeline)
│   └── app/
│       ├── api/routes.py        # /api/analyze · /api/health · /api/labels
│       ├── core/config.py       # settings (upload limits, version)
│       ├── services/            # pipeline_service (singleton, lazy load)
│       ├── schemas/             # Pydantic request/response models
│       └── main.py              # app factory + CORS
├── models/                       # the three layers + orchestration
│   ├── classification/          # DenseNet121 / EfficientNet / ViT: model, dataset, train, infer
│   ├── explainability/          # Grad-CAM, Score-CAM, explainer, overlay rendering
│   ├── report_generation/       # LLM (Claude) + offline-template generator, prompts
│   ├── common/                  # constants, config, preprocessing (DICOM ingest)
│   └── pipeline.py              # Image → Prediction → Evidence → Report
├── evaluation/                   # AUROC/F1, localization IoU, ablation, multi-seed
│   ├── evaluate.py              # predictive quality + bootstrap CIs
│   ├── evaluate_localization.py # pointing game / IoU vs. NIH boxes
│   ├── ablation.py              # classification-only vs. +localization vs. full
│   ├── aggregate_seeds.py       # mean ± std across training seeds
│   └── metrics.py · repro.py    # metric defs + reproducibility stamping
├── results/                      # committed example outputs (see results/README.md)
│   ├── output_sheets/           # per-image prediction CSV + structured findings JSON
│   └── evaluation/              # eval / localization / ablation / aggregate snapshots
├── datasets/                     # dataset docs + prep scripts (+ tiny synthetic sample sets)
│   └── samples/                 # committed synthetic studies (one DICOM each):
│       ├── chestxray14/         #   24 chest studies (NIH layout)
│       ├── brain_mri/           #   12 brain-MRI studies (Modality=MR)
│       └── head_ct/             #   12 head-CT studies (Modality=CT)
├── notebooks/                    # data exploration + pipeline walkthrough
├── tests/                        # torch-free unit tests (metrics, ablation, repro, …)
├── docs/                         # architecture · setup · deployment · API reference
│   └── images/architecture.svg  # the system diagram
├── paper/                        # LaTeX draft (main.tex + figures/) + build notes
├── demo/                         # CLI demo (run_demo.py) + generated assets/
├── configs/default.yaml          # single source of tunables (backbone, CAM, report backend)
├── docker-compose.yml            # backend :8000 + frontend :3000
├── Makefile · requirements.txt
└── README.md
```

## Datasets

MIRROR is multi-modality; each modality maps to a public benchmark:

| Modality | Primary dataset | Taxonomy |
| --- | --- | --- |
| **Chest X-ray** | NIH **ChestX-ray14** (112,120 images) | 14 disease categories |
| **Brain MRI** | **Brain Tumor MRI Dataset** (+ routine neuro findings) | 11 findings |
| **Head CT** | **RSNA Intracranial Hemorrhage** (+ routine acute-CT findings) | 11 findings |

**Other secondary sources:** RSNA Pneumonia Detection Challenge, MIMIC-CXR (images
+ reports), COVID-19 Radiography Database.

None are redistributed here. Tiny **synthetic** stand-ins ship under
`datasets/samples/` (`chestxray14/`, `brain_mri/`, and `head_ct/`, each with one
DICOM carrying the correct `Modality` tag), so the demo, the DICOM auto-routing,
loaders, and smoke tests run with zero downloads. See
[`datasets/README.md`](datasets/README.md) for the expected layout, the NIH
downloader (`download_chestxray14.py`), licensing notes, and the sample generators
(`make_synthetic_samples.py`, `make_synthetic_neuro_samples.py`).

## Configuration

Everything tunable lives in [`configs/default.yaml`](configs/default.yaml): swap
the backbone, switch Grad-CAM ↔ Score-CAM, or change the report backend without
touching code:

```yaml
model:   { backbone: densenet121 }      # or efficientnet_b0 / vit_b_16
explain: { method: gradcam }            # or scorecam
report:  { provider: template }         # or anthropic (needs ANTHROPIC_API_KEY)
```

## Train & evaluate

```bash
# Train (requires ChestX-ray14 locally)
python -m models.classification.train --config configs/default.yaml

# Evaluate prediction quality → evaluation/results/. Reports AUROC + AUPRC,
# sensitivity/specificity/PPV/NPV at the operating point, and calibration
# (Brier, ECE), all with bootstrap 95% CIs. Use --modality for brain MRI / CT.
python -m evaluation.evaluate --config configs/default.yaml \
    --checkpoint models/checkpoints/densenet121_best.pt --modality "chest X-ray"

# Evaluate explanation quality (pointing game / localization IoU) against the
# NIH ground-truth boxes (BBox_List_2017.csv) → JSON in evaluation/results/
python -m evaluation.evaluate_localization --config configs/default.yaml \
    --checkpoint models/checkpoints/densenet121_best.pt

# Ablation: classification-only baseline vs. +localization vs. full MIRROR.
# Folds the JSON above into one comparison table + a latency profile.
python -m evaluation.ablation --config configs/default.yaml \
    --prediction-results evaluation/results/eval_densenet121.json \
    --localization-results evaluation/results/loc_densenet121_gradcam.json

# Robustness across training seeds: train with --seed {0,1,2}, evaluate each,
# then aggregate to mean ± std → evaluation/results/aggregate_<backbone>.json
python -m evaluation.aggregate_seeds evaluation/results/eval_seed*.json
```

The harnesses answer the project's two questions side by side. `evaluate.py` scores
*what* the model predicts, a clinical-grade panel of per-label and macro AUROC/AUPRC,
macro F1, operating-point sensitivity / specificity / PPV / NPV, and calibration
(Brier, ECE), all with bootstrap 95% CIs. `evaluate_localization.py` scores *whether
the highlighted evidence is in the right place* against the ~984 NIH lesion boxes
(pointing game, IoU). `ablation.py` builds the baseline comparison the research
question names (classification-only vs. +localization vs. full MIRROR); because
layers 2-3 are post-hoc, the predictive column is identical across rows, so
interpretability is added at no predictive cost. `evaluation/results/` is
git-ignored; the committed snapshot in [`results/`](results/) holds the real measured
results the paper reports (`results/chestmnist/`, `results/synthetic_validation/`)
plus format-only illustrative examples. Details:
[`evaluation/README.md`](evaluation/README.md).

## Paper

**Status: finalized for submission (medRxiv, Radiology).** The paper is a complete,
all-measured draft: every number is a real result from this repo's code, with no
placeholder or pending values. LaTeX source: [`paper/main.tex`](paper/main.tex).
Compiled PDF snapshots (v1 through v6, newest last) are in
[`paper/pdf-drafts/`](paper/pdf-drafts/), with **v6 the current submission version**.

Measured highlights: DenseNet-121 reaches macro AUROC **0.729** (95% CI [0.718,
0.738]) on ChestMNIST with a clinically sensible per-label ordering; the ablation
verifies the interpretability layers are strictly post-hoc (maximum probability
change 0), so they add no predictive cost, only a bounded ~40 ms Grad-CAM pass; and a
synthetic control (0.917 vs 0.557 AUROC on signal vs no-signal labels) confirms the
metrics measure real discrimination. Framed honestly for a clinical readership:
ChestMNIST is a downsampled, reduced-budget *systems demonstration*, not a diagnostic
benchmark; grounding is *finding-level* (the descriptive prose is not pixel-verified);
and quantitative localization against the NIH boxes is future work.

See [`paper/README.md`](paper/README.md) for the section map, build steps
(Overleaf or `pdflatex main`), and how the result tables regenerate from the
evaluation harnesses.

## Documentation

- [`docs/architecture.md`](docs/architecture.md): how the layers connect + the
  local/Vercel deployment topology.
- [`docs/setup.md`](docs/setup.md): installation and running locally.
- [`docs/deployment.md`](docs/deployment.md): deploy a live public site on Vercel.
- [`docs/deployment-showcase.md`](docs/deployment-showcase.md): how the v1.1.0
  live deployment works, with an analyzed sample session.
- [`docs/api.md`](docs/api.md): REST endpoints and payloads.
- [`results/README.md`](results/README.md): committed example outputs and metric
  snapshots.
- [`paper/README.md`](paper/README.md): the workshop/preprint draft, its section
  map, and how result tables are generated.
- [`CONTRIBUTING.md`](CONTRIBUTING.md): development guidelines.

## Safety, ethics, and limitations

- **Not for clinical use.** Outputs are drafts for research into explainability
  and trust; they require clinician verification.
- **No PHI in version control.** Raw images, weights, and results are git-ignored.
- **Grounded by design.** The language layer never sees pixels, only structured
  evidence, so it cannot invent findings the classifier didn't produce.
- **Honest about uncertainty.** Predictions are probabilities; below-threshold
  findings are reported as pertinent negatives, not silently dropped.

## License

MIT. See [`LICENSE`](LICENSE), including the research-use medical disclaimer.
