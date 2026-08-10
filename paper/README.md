# Paper

Scaffold for the MIRROR write-up. Drafts and figures live here; nothing in this
directory is required to run the system.

## Current draft

[`main.tex`](main.tex) is a two-column preprint draft, capped at **8 pages**,
submitted to **SSRN** on August 6, 2026 and live as
[preprint abstract 7245078](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7245078)
(CC BY-NC-ND). It first went to medRxiv (Radiology), which declined the work as a
technical development rather than biomedical research.

The current revision is **v9** in [`pdf-drafts/`](pdf-drafts/), an accuracy pass
over v8 with no new experiments. What changed, in short: the post-hoc invariance
check is now a regression test rather than a headline claim, because a maximum
probability delta of zero is true by construction and not a finding; the ablation
table no longer prints one AUROC measurement in three rows as though it were
three results; both screenshot figures state that the **hosted vision-LLM
engine** produced them, and the paper says plainly that the grounding guarantee
holds for the local PyTorch stack only, since the hosted engine reads pixels at
every stage. Two things are newly reported: the operating point, where 11 of 14
labels emit no positive prediction at all at threshold 0.5, and a calibration
baseline showing that a constant-prevalence predictor scores macro Brier 0.047
against the model's 0.045. [`REVISION-NOTES-v9.md`](REVISION-NOTES-v9.md) maps
every changed claim to the file that justifies it. The title block, author blocks,
GIST affiliation, acknowledgments, and declarations are unchanged from v8.

Everything textual lives in the one file: literature review, architecture, and
experimental setup, an inline TikZ architecture figure, an inline `pgfplots`
per-label AUROC graph, the result tables, and an embedded 16-source bibliography.
The only external assets are the three UI screenshots in [`figures/`](figures/)
(`ui-predictions.png`, `overlay-consolidation.png`, `report-findings.png`);
upload that folder alongside `main.tex`. Each screenshot is guarded by
`\IfFileExists`, so the document still compiles (showing a placeholder box) if the
images are missing.

Every number in the draft traces to a JSON committed under
[`../results/`](../results/) or to [`calibration_baseline.py`](calibration_baseline.py),
which recomputes the no-skill calibration baseline from the committed ChestMNIST
evaluation. Nothing from `results/evaluation/` (the hand-written format
placeholders) is cited. Quantitative results are **chest X-ray only**; the
brain-MRI and head-CT paths are registered, routed, and exercised by the test
suite but have no trained checkpoints, so the paper makes no predictive claim on
them.

[`abstract-ssrn.txt`](abstract-ssrn.txt) holds the abstract as plain text,
matching the PDF, ready to paste into SSRN's metadata field during revision.

### Build

Compiles in [Overleaf](https://www.overleaf.com/) (or any TeX install) in a
**single pdfLaTeX pass**; the bibliography is embedded as a `thebibliography`
environment, so there is no separate `bibtex` step. Create an Overleaf project,
upload `main.tex` **and the `figures/` folder**, and hit *Recompile*, or locally:

```bash
cd paper
pdflatex main
```

After compiling, check that the output is at most 8 pages, then save the PDF
into [`pdf-drafts/`](pdf-drafts/) as the next `MIRROR_Paper_Draft_vN.pdf` and
add a row to the version table there.

## Extending the draft

To externalize the bibliography or figures as the paper grows:

```
paper/
├── main.tex            # manuscript source (self-contained draft, provided)
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
  `evaluation/ablation.py` → `evaluation/results/ablation_<backbone>.json`. Read the
  AUROC field here as **one** measurement, not three: `assemble_table()` copies a
  single `macro_auroc` into every row because the post-hoc layers cannot alter a
  prediction. The measurements that actually vary across rows are `max_prob_delta`
  (a regression check, expected to be exactly 0) and the per-stage latency.

Qualitative figures (saliency overlays) come from `demo/run_demo.py` or notebook
`02_pipeline_walkthrough.ipynb`.

> The "Potential Paper Title" and "Expected Deliverables" sections from the
> original project overview were intentionally left out of this scaffold per the
> build brief.
