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
measure with two parallel harnesses:

- `evaluation/evaluate.py` — **prediction** quality: per-label/macro AUROC, F1.
- `evaluation/evaluate_localization.py` — **explanation** quality: scores each
  Grad-CAM/Score-CAM map against NIH's lesion boxes (`BBox_List_2017.csv`) with
  pointing-game accuracy, mean IoU, and localization accuracy at an IoU
  threshold. This is what turns "Layer 2 highlights evidence" from a design claim
  into a measured one.
- `evaluation/ablation.py` — the **baseline comparison**: classification-only vs.
  +localization vs. full MIRROR, in one table. It confirms the predictions are
  unchanged by the later layers (so AUROC/F1 hold across conditions) and profiles
  each layer's latency — framing MIRROR's contribution as interpretability at no
  predictive cost.

Predictive numbers carry bootstrap 95% CIs (`metrics.bootstrap_cis`) and can be
aggregated across training seeds via `evaluation/aggregate_seeds.py` (mean ± std);
every results JSON stamps a `reproducibility` block (`evaluation/repro.py`: seed,
git commit, library versions) so a number traces back to the exact code that made
it.

See [`evaluation/README.md`](../evaluation/README.md) for all four harnesses.

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
notebooks consume. The two later layers are toggleable — `analyze(localize=...,
report=...)` — which recovers the ablation conditions (classification-only,
+localization, full) and records per-stage timings in `meta['timings_ms']`. The
defaults run everything, so serving is unchanged.

## Serving

MIRROR is served by a thin client over a pluggable inference engine. The
frontend is identical in every deployment; a single environment variable
(`NEXT_PUBLIC_API_URL`) decides which engine answers an analyze request.

- **Frontend** (`frontend/`) — a Next.js "reading-room" UI: upload, a film viewer
  with an evidence-overlay toggle, the per-label predictions, and the draft
  report. It POSTs the image to `…/api/analyze` and renders whatever comes back
  in the shared response contract (`modality`, `backbone`, `explain_method`,
  `report`, `report_backend`, `findings[]`, `meta`). Each finding carries a
  probability, a present/absent flag, a plain-English location, and **either** a
  rendered Grad-CAM overlay PNG **or** a normalized bounding box — the viewer
  draws whichever is present.
- **Backend** (`backend/`) — FastAPI exposes `/api/analyze`, `/api/health`,
  `/api/labels`. It wraps the real `MirrorPipeline`; the pipeline loads lazily on
  first request and is reused across calls. This is the engine for local
  development and any host with a Python/PyTorch runtime.

## Deployment topology

Two interchangeable engines satisfy the same contract, so the UI never changes:

| | Local full stack | Hosted (Vercel) |
| --- | --- | --- |
| Inference engine | FastAPI + real `MirrorPipeline` (PyTorch) | Next.js serverless route, Claude **vision** |
| Classification | DenseNet121 / EffNet-B0 / ViT-B/16 | Claude scores the 14 labels from the image |
| Localization | Grad-CAM / Score-CAM heatmap PNG | Claude returns a bounding box per finding |
| Report | LLM **or** offline template | Claude (same evidence-grounded prompt shape) |
| Inputs | PNG/JPEG/BMP/WEBP **+ DICOM** | PNG/JPEG/WEBP |
| `NEXT_PUBLIC_API_URL` | `http://localhost:8000` | *(unset → same-origin route)* |
| Needs | Python, ~6 GB weights/deps | one env var: `ANTHROPIC_API_KEY` |

The hosted route lives at
[`frontend/app/api/analyze/route.ts`](../frontend/app/api/analyze/route.ts). It
exists because the PyTorch pipeline cannot fit Vercel's serverless constraints
(no GPU, a 250 MB bundle ceiling, cold-start model loads), so the public demo
substitutes Claude's vision model as a drop-in engine that emits the identical
JSON. With no API key it returns a clearly-labelled deterministic demo result, so
the deployed site never hard-fails. See [`deployment.md`](deployment.md) for the
step-by-step Vercel guide and the one-click deploy button.

Because both engines honour the same contract, a finding localized as a Grad-CAM
PNG (local) and one localized as a bounding box (hosted) render through the same
film viewer with the same overlay toggle — the *interface* MIRROR is testing
(predict → show evidence → explain) is preserved regardless of which engine
produced the evidence.
