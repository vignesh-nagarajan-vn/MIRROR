# Architecture

MIRROR turns a single radiograph into a reviewable diagnostic draft by chaining
three complementary layers. The defining design choice is that each layer's
output becomes the *grounded input* to the next, so the final report can always
be traced back to a probability and a saliency region.

```
                    ┌─────────────────────────────┐
   radiograph  ───▶ │  1. CLASSIFICATION          │  CNN / ViT backbone
                    │     DenseNet121 · EffNet-B0  │  → 14 per-label
                    │     · ViT-B/16               │    probabilities
                    └──────────────┬──────────────┘
                                   │ predictions
                    ┌──────────────▼──────────────┐
                    │  2. EVIDENCE LOCALIZATION    │  Grad-CAM / Score-CAM
                    │     hooks the target layer   │  → heatmap + region
                    │     for each positive label  │    (centroid, bbox)
                    └──────────────┬──────────────┘
                                   │ structured evidence
                    ┌──────────────▼──────────────┐
                    │  3. CLINICAL REASONING       │  LLM (Claude) or
                    │     prompts over evidence    │  offline template
                    │     only — never the pixels  │  → FINDINGS / IMPRESSION
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │     Human-readable report    │
                    └─────────────────────────────┘
```

## Why this differs from classify-only systems

A traditional medical-imaging model stops at `Image → Prediction`. MIRROR adds
two layers whose explicit job is to communicate *why* a prediction was made:

- **Localization** answers "where in the image?" via class activation maps.
- **Reasoning** answers "what does this mean clinically?" by composing the
  predictions and their locations into prose a clinician can scan.

The research question is whether these layers improve interpretability and trust
without degrading predictive performance — which `evaluation/` is set up to
measure (AUROC/F1 for prediction; pointing-game/IoU for localization).

## Layer 1 — Classification (`models/classification/`)

- `model.py` — a factory over DenseNet121, EfficientNet-B0, and ViT-B/16 with a
  14-way multi-label head (sigmoid applied at loss/inference time).
- `dataset.py` — ChestX-ray14 multi-hot label encoding.
- `train.py` — `BCEWithLogitsLoss`, AdamW, cosine schedule; checkpoints on best
  validation macro-AUROC.
- `infer.py` — a stateful `Classifier` used by the pipeline.

## Layer 2 — Explainability (`models/explainability/`)

- `gradcam.py` — forward/backward hooks on a target layer; ViT support via a
  token-to-grid reshape.
- `scorecam.py` — gradient-free, perturbation-based maps.
- `explainer.py` — resolves the right target layer per backbone, renders the
  overlay, and derives a centroid/bbox so the region can be *named* in words.
- `overlay.py` — colormap + alpha-blend rendering to PNG.

## Layer 3 — Report generation (`models/report_generation/`)

- `prompts.py` — a system prompt and an evidence-grounded user prompt. The LLM
  never sees pixels; it reasons over the structured evidence, which keeps it
  honest and the output auditable.
- `generator.py` — Anthropic (Claude) backend with a deterministic offline
  template fallback so the system always produces a coherent report.

## Orchestration (`models/pipeline.py`)

`MirrorPipeline.analyze()` runs all four stages and returns one
`AnalysisResult` (predictions + overlays + report) that both the API and the
notebooks consume.

## Serving

- **Backend** (`backend/`) — FastAPI exposes `/api/analyze`, `/api/health`,
  `/api/labels`. The pipeline loads lazily and is reused across requests.
- **Frontend** (`frontend/`) — a Next.js "reading-room" UI: upload, film viewer
  with an evidence-overlay toggle, predictions, and the draft report.
