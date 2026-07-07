# Paper

Scaffold for the MIRROR write-up. Drafts and figures live here; nothing in this
directory is required to run the system.

## Current draft

[`main.tex`](main.tex) is a **single, self-contained** draft targeting a short
(5–8 page) medical-imaging workshop paper / arXiv preprint. It merges the written
literature review, architecture, and experimental-setup sections with an inline
TikZ architecture figure, the three result tables, and an embedded bibliography
(20 sources) — everything lives in the one file.

The results-independent sections (introduction, literature review, architecture,
experimental setup, ethics and limitations) are complete. The result numbers are
**illustrative, literature-scale placeholders** pulled from the committed
`results/evaluation/*.json` snapshots (they document each harness's output
*format*, not a measured benchmark). Everything still blocked on a trained model
is wrapped in a red `\pending{...}` note — **search the source for `PENDING` to
find every gap.**

### Build

It compiles as-is in [Overleaf](https://www.overleaf.com/) (or any TeX install)
with a **single pdfLaTeX pass** — the bibliography is embedded as a
`thebibliography` environment, so there is no separate `bibtex` step and no
"undefined references" on first compile. Drop `main.tex` into a blank Overleaf
project and hit *Recompile*, or locally:

```bash
cd paper
pdflatex main
```

> **Before making any performance claim**, replace the placeholder numbers with
> real values: train on ChestX-ray14, run the `evaluation/` harnesses, and copy
> the figures from the resulting JSON into the tables in `main.tex`.

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
