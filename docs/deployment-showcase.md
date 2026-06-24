# v1.1.0 Deployment Showcase

This document walks through the **live v1.1.0 deployment** of MIRROR at
**[mirror-ten-jet.vercel.app](https://mirror-ten-jet.vercel.app/)**: how it is
built, how a request flows end to end, and a finding-by-finding read of one real
session.

> **Research prototype, not for clinical use.** Everything below is a draft
> produced by an AI system and would require verification by a licensed
> radiologist. The sample image is a public demonstration radiograph.

## How the v1.1.0 deployment works

The hosted site is the **Next.js frontend deployed on Vercel**, with no separate
backend server. The PyTorch pipeline (DenseNet/ViT classifier + Grad-CAM) cannot
run inside Vercel's serverless limits (no GPU, a 250 MB bundle ceiling,
cold-start model loads), so the deployment swaps in a different inference engine
that satisfies the **same response contract** the UI already consumes.

```
Browser (Next.js UI)
   │  multipart POST  (image, modality, indication)
   ▼
/api/analyze   ← Next.js serverless route (frontend/app/api/analyze/route.ts)
   │  image + structured prompt, forced tool-call
   ▼
Claude vision  (claude-haiku-4-5)
   │  returns: 14 per-label probabilities, a bbox per positive
   │           finding, and a FINDINGS / IMPRESSION report
   ▼
Normalized AnalysisResponse  →  rendered as predictions + overlays + report
```

Key properties of this engine:

- **Same JSON contract as local.** The route returns `modality`, `backbone`,
  `explain_method`, `report`, `report_backend`, `findings[]`, and `meta`, so the
  identical React components render it. Locally those fields come from the real
  `MirrorPipeline`; here they come from Claude. The frontend code does not change.
- **Localization as bounding boxes.** Instead of a rendered Grad-CAM heatmap PNG,
  Claude returns a normalized `[x, y, w, h]` box per positive finding, which the
  film viewer draws over the image (accounting for `object-fit: contain`
  letterboxing so it stays aligned on non-square uploads).
- **Secrets stay server-side.** `ANTHROPIC_API_KEY` is read only inside the
  serverless function, never shipped to the browser.
- **Never hard-fails.** With no API key the route returns a clearly-labelled
  deterministic demo result, so the site always renders.

The footer of the report panel surfaces the active engine:
`backbone: claude-vision (claude-haiku-4-5) · explain: bbox · report backend: anthropic`.

> **Model note.** The default model is **`claude-haiku-4-5`**, chosen to preserve
> tokens and cut cost: Haiku 4.5 is **$1 / $5** per million input/output tokens
> versus Sonnet 4.6's **$3 / $15** — about **3× cheaper** (roughly one-third the
> cost) per analysis. Override it with the `ANTHROPIC_MODEL` env var. The sample
> screenshots below were captured on the earlier `claude-sonnet-4-6` default, so
> their report footer still reads `claude-sonnet-4-6`.

For the architecture and the local-vs-hosted topology table, see
[`architecture.md`](architecture.md); for the deploy steps, see
[`deployment.md`](deployment.md).

> **What this is and isn't.** On the hosted engine the probabilities are Claude's
> visual estimates, and the boxes are Claude's localization, **not** the output of
> a trained, calibrated DenseNet classifier or a Grad-CAM saliency map. For
> reproducible, benchmarked predictive/localization numbers, train on NIH
> ChestX-ray14 and run the local stack and the harnesses in `evaluation/`.

## Sample session

Input: a frontal chest radiograph uploaded with modality **chest X-ray** and the
clinical indication **"productive cough, 3 days."**

### 1. Input and per-label predictions

![Study input and predictions](images/deployment/mirror-sample1-input-predictions-ui.png)

All 14 ChestX-ray14 labels are scored. Three clear the 50% reporting threshold:

| Finding | Probability | Localized to |
| --- | --- | --- |
| **Pneumonia** | 78% | right lower lobe predominant, bilateral lower zones |
| **Infiltration** | 72% | right lower and middle lung zones, left lower lung zone |
| **Consolidation** | 65% | right lower lobe and peri-hilar right lung |
| Edema | 20% | below threshold |
| Atelectasis | 18% | below threshold |
| Cardiomegaly | 15% | below threshold |
| Effusion | 12% | below threshold |
| Nodule | 10% | below threshold |
| Pleural Thickening | 9% | below threshold |
| Emphysema | 8% | below threshold |
| Mass | 7% | below threshold |
| Fibrosis | 6% | below threshold |
| Pneumothorax | 3% | below threshold |
| Hernia | 2% | below threshold |

The three positives are clinically coherent: pneumonia, infiltration, and
consolidation are overlapping descriptors of the same airspace-filling process,
and they cluster in the right lower zone, which matches the productive-cough
indication. Everything else stays low, so the below-threshold labels read as
pertinent negatives rather than noise.

### 2. Evidence localization

Each positive finding is drawn as a labelled box on the film; the chips under the
viewer switch between them, and the **evidence overlay** toggle fades the boxes
in and out.

| Consolidation (right lower lobe) | Infiltration (bilateral lower zones) |
| :---: | :---: |
| ![Consolidation overlay](images/deployment/mirror-sample1-diagnosis-consolidation.png) | ![Infiltration overlay](images/deployment/mirror-sample1-diagnosis-infiltration.png) |

The Consolidation box is tight on the right lower lobe, the densest region of the
film. The Infiltration box is larger and spans both lower zones, reflecting the
more diffuse pattern that label describes. This is exactly the behaviour MIRROR
is built to demonstrate: a prediction is not just a number, it is tied to a
*place* a reviewer can scan.

### 3. Draft clinical report

| FINDINGS | IMPRESSION |
| :---: | :---: |
| ![Report findings](images/deployment/mirror-sample1-output-analysis-top.png) | ![Report impression](images/deployment/mirror-sample1-output-analysis-bottom.png) |

The report is structured into `FINDINGS` and `IMPRESSION`, and every statement
traces back to the evidence above:

- **FINDINGS** describes the right-lower-lobe consolidation/infiltration in high
  confidence, notes the more subtle bilateral lower-zone infiltrates, and then
  walks the pertinent negatives: normal cardiac silhouette (cardiothoracic ratio
  < 0.5), clear pleural spaces, clear upper/middle zones, midline mediastinum,
  intact bony thorax, and no pneumothorax.
- **IMPRESSION** prioritizes: right lower lobe consolidation as the dominant
  finding and, *given the 3-day productive cough*, frames it as most consistent
  with community-acquired bacterial pneumonia; flags possible bilateral
  involvement; confirms no effusion, pneumothorax, or cardiomegaly; and
  recommends clinical correlation plus a follow-up radiograph after antibiotic
  therapy.
- It ends with the mandatory **AI-GENERATED DRAFT** disclaimer.

Two things to note about *why* the report reads this way. First, it incorporates
the supplied indication ("productive cough, 3 days") into the impression, which
is how a real reasoning step narrows a differential. Second, it does not invent
findings: the negatives it lists are the labels that scored low, so the prose
stays grounded in the structured evidence rather than free-associating.

## Reproducing this

1. Open [mirror-ten-jet.vercel.app](https://mirror-ten-jet.vercel.app/).
2. Drop in a PNG/JPEG/WEBP chest radiograph.
3. Set the modality and (optionally) a clinical indication.
4. Click **Run analysis**.

Results vary run to run because the engine is a vision-language model, not a
fixed-weight classifier. For deterministic, benchmarked output, run the local
PyTorch stack ([`setup.md`](setup.md)) and the evaluation harnesses
([`../evaluation/README.md`](../evaluation/README.md)).
