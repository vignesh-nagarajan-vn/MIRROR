# Paper

Scaffold for the MIRROR write-up. Drafts and figures live here; nothing in this
directory is required to run the system.

## Current draft

[`main.tex`](main.tex) is a two-column workshop/preprint draft (5–8 pages)
targeting a top medical-imaging venue. Everything textual lives in the one file:
the literature review, architecture, and experimental-setup sections, an inline
TikZ architecture figure, two inline `pgfplots` result graphs, the result
tables, and an embedded 20-source bibliography. The only external assets are the
three UI screenshots in [`figures/`](figures/) (`ui-predictions.png`,
`overlay-consolidation.png`, `report-findings.png`) — upload that folder alongside
`main.tex`. Each screenshot is guarded by `\IfFileExists`, so the document still
compiles (showing a red placeholder box) if the images are missing.

The draft describes MIRROR as the **multi-modality** system the codebase now is:
the modality registry (`tab:modalities`) is documented as an implemented, tested
contribution (black), while quantitative results remain **chest X-ray only** because
that is the modality with public labels and lesion boxes. The predictive results now
include the full clinical-reader panel — discrimination (AUROC/AUPRC/F1), the
operating-point table (`tab:operating`: sensitivity/specificity/PPV/NPV with
support), and calibration (Brier/ECE) — matching what `evaluation/evaluate.py`
writes. Every one of those numbers stays **red** (illustrative placeholder) until a
trained checkpoint replaces it; the brain-MRI and head-CT paths are described as
implemented-but-not-yet-benchmarked, not as scaffolding.

### Draft color convention (black = done, red = pending)

Per the current working style, the draft encodes its own to-do list in color:

- **Black** text is finalized/complete content.
- **Red** text is filler awaiting a trained model — every placeholder number,
  chart datum, and `\pending{...}` note. The result tables and both graphs render
  their numbers in red because they are **illustrative, literature-scale
  placeholders** from the committed `results/evaluation/*.json` snapshots (output
  *format*, not a measured benchmark).

**Search the source for `PENDING` to find every gap.** When a real value replaces
a red one, set it black.

### Build

Compiles in [Overleaf](https://www.overleaf.com/) (or any TeX install) in a
**single pdfLaTeX pass** — the bibliography is embedded as a `thebibliography`
environment, so there is no separate `bibtex` step. Create an Overleaf project,
upload `main.tex` **and the `figures/` folder**, and hit *Recompile*, or locally:

```bash
cd paper
pdflatex main
```

> **Before making any performance claim**, replace the red placeholder numbers
> with real values: train on ChestX-ray14, run the `evaluation/` harnesses, and
> copy the figures from the resulting JSON into the tables and graphs in
> `main.tex` (then set them black).

## Extending the draft

To externalize the bibliography or figures as the paper grows:

```
paper/
├── main.tex            # manuscript source (self-contained draft — provided)
├── references.bib      # optional: move the embedded \thebibliography here + bibtex
├── figures/            # optional: exported figures (architecture, qualitative overlays)
└── tables/             # optional: \input-able tables generated from results JSON
```

The result tables in `main.tex` map one-to-one onto the evaluation harnesses, so
they regenerate directly from JSON:

- **Prediction table** (per-label / macro AUROC, macro F1) from
  `evaluation/evaluate.py` → `evaluation/results/eval_<backbone>.json`. Report each
  number with its **bootstrap 95% CI** (`*_ci` fields), and where you have multiple
  seeds, the **mean ± std** from `evaluation/aggregate_seeds.py`
  (`aggregate_<backbone>.json`). The `reproducibility` block (seed, git commit)
  belongs in an appendix or footnote so results are regenerable.
- **Localization table** (pointing game, mean IoU, localization accuracy over the
  8 boxed pathologies) from `evaluation/evaluate_localization.py` →
  `evaluation/results/loc_<backbone>_<method>.json`. This substantiates the
  explainability claim, so it belongs beside the prediction table.
- **Ablation table** (classification-only vs. +localization vs. full MIRROR) from
  `evaluation/ablation.py` → `evaluation/results/ablation_<backbone>.json`. This is
  the baseline comparison the research question names: prediction metrics are
  unchanged by the added layers (interpretability at no predictive cost), while the
  localization/report capabilities and their latency appear only in the relevant
  rows.

Qualitative figures (saliency overlays) come from `demo/run_demo.py` or notebook
`02_pipeline_walkthrough.ipynb`.

> The "Potential Paper Title" and "Expected Deliverables" sections from the
> original project overview were intentionally left out of this scaffold per the
> build brief.
