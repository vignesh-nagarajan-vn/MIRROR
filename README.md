<div align="center">

# MIRROR

**Multimodal Intelligent Radiology Reasoning and Observation Reporter**

*An explainable medical-AI system that reads radiological images, localizes its
own evidence, and writes a clinician-style draft report.*

</div>

---

> ⚠️ **Research prototype — not a medical device.** MIRROR has not been reviewed
> or cleared by any regulatory body. Every output is a draft that must be
> verified by a licensed radiologist. Do not use it for diagnosis or treatment.

## What MIRROR does

Most medical-imaging models stop at a prediction. MIRROR adds the two layers that
make a prediction *trustworthy and usable*: it shows **where** the evidence is and
explains **what it means** in plain clinical language.

It analyzes a radiograph (chest X-ray today; CT and brain MRI are scaffolded),
identifies potential abnormalities, highlights the diagnostic evidence with
saliency overlays, and generates a structured natural-language report.

```
Image → Prediction → Evidence Localization → Clinical Reasoning → Human-Readable Report
```

![Architecture](docs/images/architecture.svg)

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

1. **Radiological image understanding** — a CNN/ViT classifier.
2. **Visual evidence localization** — Grad-CAM / Score-CAM saliency.
3. **Natural-language clinical report generation** — an LLM (or offline template)
   that reasons *only* over the structured evidence above.

The result not only predicts abnormalities but communicates **why** the
prediction was made and **how** it relates to potential clinical findings — and
every sentence in the report traces back to a probability and a saliency region.

## Quickstart

```bash
# 1. Install
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Run the full pipeline on one image (no checkpoint or API key needed)
python -m demo.run_demo path/to/chest_xray.png
```

That prints predictions and a draft report, and writes Grad-CAM overlays to
`demo/assets/`. It runs on ImageNet-pretrained weights with the offline template
report backend, so it works anywhere.

### Full stack (web UI)

```bash
# Backend
cd backend && uvicorn app.main:app --reload --port 8000

# Frontend (new terminal)
cd frontend && npm install && cp .env.local.example .env.local && npm run dev
```

Open http://localhost:3000 — drop in a radiograph, toggle the evidence overlay,
read the draft report. Full instructions in [`docs/setup.md`](docs/setup.md).

## Repository layout

```
mirror/
├── frontend/                 # Next.js "reading-room" UI
├── backend/                  # FastAPI service (lazy-loads the pipeline)
│   └── app/{api,core,services,schemas}
├── models/                   # the three layers + orchestration
│   ├── classification/       # DenseNet121 / EfficientNet / ViT, train + infer
│   ├── explainability/       # Grad-CAM, Score-CAM, overlay rendering
│   ├── report_generation/    # LLM + offline-template report generator
│   ├── common/               # constants, config, preprocessing
│   └── pipeline.py           # Image → Prediction → Evidence → Report
├── notebooks/                # data exploration + pipeline walkthrough
├── datasets/                 # dataset docs + prep scripts (no data committed)
├── evaluation/               # AUROC/F1 + localization metrics
├── paper/                    # write-up scaffold
├── docs/                     # architecture, setup, API reference
├── demo/                     # CLI demo + assets
├── configs/default.yaml      # single source of tunables
└── README.md
```

## Technology stack

| Area | Choice |
| --- | --- |
| Deep learning | PyTorch |
| Backbones | DenseNet121 · EfficientNet-B0 · ViT-B/16 |
| Explainability | Grad-CAM · Score-CAM |
| Report generation | LLM (Claude) with offline template fallback |
| Backend | FastAPI |
| Frontend | Next.js (TypeScript) |
| Database | PostgreSQL / Supabase *(optional; for persisting studies — see below)* |

> The database is optional and not required to run the pipeline. The current API
> is stateless; persisting studies/reports to PostgreSQL or Supabase is a natural
> extension and the schema hook lives behind the service layer.

## Datasets

**Primary:** NIH **ChestX-ray14** — 112,120 chest X-rays, 14 disease categories;
ideal for initial development and benchmarking.

**Secondary:** RSNA Pneumonia Detection Challenge, MIMIC-CXR (images + reports),
Brain Tumor MRI Dataset, COVID-19 Radiography Database.

None are redistributed here. See [`datasets/README.md`](datasets/README.md) for
the expected layout, licensing notes, and prep scripts.

## Configuration

Everything tunable lives in [`configs/default.yaml`](configs/default.yaml) — swap
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

# Evaluate a checkpoint → JSON in evaluation/results/
python -m evaluation.evaluate --config configs/default.yaml \
    --checkpoint models/checkpoints/densenet121_best.pt
```

## Potential contributions

- An end-to-end multimodal radiology analysis pipeline.
- An explainable-AI framework for medical imaging (Grad-CAM / Score-CAM).
- Automated clinician-style report generation grounded in model evidence.
- An evaluation of interpretability versus predictive performance.
- An open-source benchmark for combining computer vision with LLM-based reasoning
  in healthcare.

## Documentation

- [`docs/architecture.md`](docs/architecture.md) — how the layers connect.
- [`docs/setup.md`](docs/setup.md) — installation and running locally.
- [`docs/api.md`](docs/api.md) — REST endpoints and payloads.
- [`CONTRIBUTING.md`](CONTRIBUTING.md) — development guidelines.

## Safety, ethics, and limitations

- **Not for clinical use.** Outputs are drafts for research into explainability
  and trust; they require clinician verification.
- **No PHI in version control.** Raw images, weights, and results are git-ignored.
- **Grounded by design.** The language layer never sees pixels — only structured
  evidence — so it cannot invent findings the classifier didn't produce.
- **Honest about uncertainty.** Predictions are probabilities; below-threshold
  findings are reported as pertinent negatives, not silently dropped.

## License

MIT — see [`LICENSE`](LICENSE), including the research-use medical disclaimer.
