# CLAUDE.md

Guidance for AI coding agents (Claude Code and others) working in this repository.
Read this first; it captures the intent, structure, and gotchas that are not
obvious from any single file.

## What MIRROR is

**MIRROR** = *Multimodal Intelligent Radiology Reasoning and Observation Reporter.*
An explainable medical-AI research prototype (GIST 2026 Summer Internship, mentored
by Sriram Venkatapathy) that takes a chest X-ray and runs a four-stage pipeline:

```
Image → Prediction → Evidence Localization → Clinical Reasoning → Human-Readable Report
```

The research question it is built to *measure* (not just demonstrate): does adding
a visual-explainability layer and an LLM report layer on top of a classifier
improve interpretability/trust **without** degrading predictive performance?

- Live demo: https://mirror-ten-jet.vercel.app/ (v1.1.0)
- License: MIT, with a research-use medical disclaimer. **Not for clinical use.**

## The core design principle (do not break this)

**The language layer never sees pixels.** The report generator reasons *only* over
structured evidence — per-label probability, present/absent flag, and a plain-English
location derived from the saliency map. This is what keeps reports auditable and
prevents the LLM from inventing findings the classifier did not produce. Any change
that feeds the raw image into the report/reasoning layer violates the project thesis.
(The hosted Vercel engine is the one intentional exception — see Serving below — and
even there the model is constrained to a structured tool schema.)

Related invariants:
- Layers 2 (localization) and 3 (report) are **post-hoc**: they never change the
  predictions. `evaluation/ablation.py` verifies this empirically
  (`predictions_invariant`, `max_prob_delta == 0`). Keep it true.
- The 14 ChestX-ray14 labels have **one source of truth**:
  [`models/common/constants.py`](models/common/constants.py) `CHESTXRAY14_LABELS`.
  The TS route [`frontend/app/api/analyze/route.ts`](frontend/app/api/analyze/route.ts)
  hard-copies this list and **must stay in the same order** (model output index i
  maps to label i).

## Architecture: the three layers + orchestration

All live under [`models/`](models/):

| Layer | Location | What it does |
| --- | --- | --- |
| 1 · Classification | [`models/classification/`](models/classification/) | Factory over DenseNet121 / EfficientNet-B0 / ViT-B/16 with a 14-way multi-label head. Sigmoid is applied at loss/inference time, **not** inside the module (logits out). |
| 2 · Explainability | [`models/explainability/`](models/explainability/) | Grad-CAM (`gradcam.py`, forward/backward hooks; ViT via token→grid reshape) or Score-CAM (`scorecam.py`, gradient-free). `explainer.py` renders the overlay PNG and derives a centroid/bbox; `describe_location()` maps the centroid to a 3×3 grid in radiology convention (patient-right = viewer-left). |
| 3 · Report generation | [`models/report_generation/`](models/report_generation/) | Two backends: `anthropic` (Claude) and `template` (offline deterministic). The anthropic backend **falls back to template on any error** (missing key, network, import). Both consume the identical evidence, so swapping backends changes fluency, not content. |

[`models/pipeline.py`](models/pipeline.py) — `MirrorPipeline.analyze()` runs all four
stages, returns one `AnalysisResult`, and records per-stage timings in
`meta['timings_ms']`. The `localize=` / `report=` flags toggle layers 2/3 and are
exactly what recovers the three ablation conditions (classification-only →
+localization → full). Only the top-k (default 3) present findings are explained,
to bound Grad-CAM compute.

Shared code in [`models/common/`](models/common/): `constants.py` (labels, ImageNet
norm, label descriptions), `config.py` (YAML-backed dataclasses + `get_anthropic_api_key`),
`preprocessing.py` (resize/normalize/denormalize), and `dicom.py` — a real DICOM
ingest that applies the modality LUT, VOI LUT, and MONOCHROME1 inversion, and lifts
only **non-PHI** technical tags.

## Serving: two engines, one JSON contract

The frontend is identical in both deployments; `NEXT_PUBLIC_API_URL` picks the engine.
Both satisfy the same response shape (`modality`, `backbone`, `explain_method`,
`report`, `report_backend`, `findings[]`, `meta`). Each finding carries **either** a
Grad-CAM `overlay_png_b64` (local) **or** a normalized `bbox` (hosted); the FilmViewer
renders whichever is present.

- **Local full stack** — FastAPI [`backend/`](backend/) wraps the real PyTorch
  `MirrorPipeline` (lazy singleton in `services/pipeline_service.py`). Endpoints:
  `/api/analyze`, `/api/health`, `/api/labels` ([`backend/app/api/routes.py`](backend/app/api/routes.py)).
  Accepts PNG/JPEG/BMP/WEBP **and native DICOM** (by extension, content-type, or
  magic bytes).
- **Hosted (Vercel)** — [`frontend/app/api/analyze/route.ts`](frontend/app/api/analyze/route.ts)
  replaces the entire PyTorch stack with **Claude vision** via forced tool-use
  (`record_analysis` tool → 14 probabilities + a bbox per present finding + report),
  because PyTorch cannot fit Vercel's serverless limits (no GPU, 250 MB bundle,
  cold starts). Accepts PNG/JPEG/WEBP (no DICOM/BMP). With **no `ANTHROPIC_API_KEY`**
  it returns a clearly-labelled deterministic `demoResult()` so the site never
  hard-fails.

Frontend is Next.js + TypeScript ([`frontend/`](frontend/)):
`UploadPanel` → `FilmViewer` (overlay/bbox toggle) → `FindingsList` → `ReportPanel`,
wired in [`frontend/app/page.tsx`](frontend/app/page.tsx); typed client in
[`frontend/lib/api.ts`](frontend/lib/api.ts).

## Evaluation harnesses (the research half)

All in [`evaluation/`](evaluation/); metric defs in `metrics.py`, provenance in
`repro.py`. Each writes a JSON that stamps a `reproducibility` block (seed, git
commit, library versions).

- `evaluate.py` — per-label & macro **AUROC**, macro **F1**, each with a
  **bootstrap 95% CI** (`bootstrap_cis`: one resample drives all metrics, so CIs are
  mutually consistent). Seeded (`--seed`, default 42).
- `evaluate_localization.py` — **pointing game**, **mean IoU**, **localization
  accuracy** at an IoU threshold, scored against NIH `BBox_List_2017.csv` (~984 boxes
  over 8 of the 14 pathologies).
- `ablation.py` — classification-only vs. +localization vs. full MIRROR in one table;
  reads the eval/loc JSON (recomputes nothing) and adds a live latency profile.
- `aggregate_seeds.py` — mean ± std across training seeds (training noise, distinct
  from the per-model bootstrap CI).

## Data and results

- **Dataset:** NIH ChestX-ray14 (primary). **Not redistributed.** A tiny
  **synthetic** stand-in ships under
  [`datasets/samples/chestxray14/`](datasets/samples/chestxray14/) (24 images, NIH
  layout, one DICOM) so the demo/loader/smoke tests run with zero downloads.
- **`evaluation/results/` is git-ignored.** The committed, curated snapshot lives in
  top-level [`results/`](results/). ⚠️ Those numbers are **illustrative,
  literature-scale placeholders, NOT measured benchmarks** — every file says so in an
  `_note` field. Do not cite them as real results or present them as MIRROR's
  measured performance.

## Working in this repo

- **Run the demo (no weights/key needed):**
  `python -m demo.run_demo datasets/samples/chestxray14/images/synth_0001.png`
  (a `.dcm` sample works too). Runs on ImageNet weights + offline template backend.
- **Tests:** `pytest` — the suite in [`tests/`](tests/) is deliberately **torch-free**
  (synthetic inputs), so it runs without a model or dataset. If you add logic, prefer
  keeping the unit-testable part importable without torch.
- **Config:** everything tunable is in [`configs/default.yaml`](configs/default.yaml)
  (backbone, `explain.method`, `report.provider`, thresholds). Prefer config over
  hardcoding.
- **Convenience:** [`Makefile`](Makefile) targets — `demo`, `backend`, `frontend`,
  `train`, `eval`, `eval-loc`, `ablation`, `aggregate`.
- **Optional deps degrade gracefully:** torch, anthropic, pydicom are all guarded by
  try/except so torch-free paths (tests, template reports, label lookups) keep working
  when they're absent. Preserve that pattern.
- **Default report/vision model:** `claude-haiku-4-5` (set in `configs/default.yaml`
  and the Vercel route; override via `ANTHROPIC_MODEL` on the hosted side). When
  touching Claude API code, use current model IDs and the current SDK shape.

## Platform / environment notes

- Primary dev environment here is **Windows** (PowerShell + Git Bash). The README's
  local instructions assume Git Bash; venv activation is `.venv/Scripts/activate`.
- Two package ecosystems: Python (`requirements.txt`, root + `backend/`) and Node
  (`frontend/package.json`).

## Safety / ethics constraints (respect these)

- Not for clinical use; outputs are drafts requiring clinician verification, and every
  report ends with an explicit AI-generated-draft disclaimer.
- No PHI in version control — raw images, weights, and real results are git-ignored;
  DICOM ingest deliberately extracts only non-PHI tags.
- Keep the "grounded by design" property (language layer sees evidence, not pixels)
  and honest uncertainty (below-threshold findings reported as pertinent negatives,
  not dropped).

## Docs map

[`docs/architecture.md`](docs/architecture.md) (layers + deployment topology),
[`docs/setup.md`](docs/setup.md), [`docs/deployment.md`](docs/deployment.md) (Vercel),
[`docs/deployment-showcase.md`](docs/deployment-showcase.md) (annotated live session),
[`docs/api.md`](docs/api.md) (REST payloads),
[`evaluation/README.md`](evaluation/README.md) and
[`results/README.md`](results/README.md) (harness + snapshot details),
[`paper/`](paper/) (write-up; `paper/main.tex` is the compilable draft).
