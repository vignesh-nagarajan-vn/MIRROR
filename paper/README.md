# Paper

Scaffold for the MIRROR write-up. Drafts and figures live here; nothing in this
directory is required to run the system.

## Current draft

A partial draft targeting a short (5 to 8 page) medical-imaging workshop paper
and arXiv preprint lives in [`main.tex`](main.tex), with
[`references.bib`](references.bib) (20 sources, including a one-page literature
review) and placeholder result tables under [`tables/`](tables/). The
results-independent sections (introduction, literature review, architecture,
experimental setup, ethics and limitations) are written; everything blocked on a
trained model is marked with a red `\pending{...}` note or lives in a placeholder
table. **Search the source for `PENDING` to find every gap.**

Build (Overleaf, or any TeX install; uses only stock packages):

```bash
cd paper
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

The draft compiles today: the placeholder tables and a boxed figure stand in
until real results and an exported architecture figure replace them.

## Suggested structure

```
paper/
├── main.tex            # manuscript source
├── references.bib      # citations (Grad-CAM, Score-CAM, ChestX-ray14, MIMIC-CXR…)
├── figures/            # exported figures (architecture, qualitative overlays)
└── tables/             # generated result tables (from evaluation/results/*.json)
```

Result tables can be generated from the JSON written by the evaluation
harnesses:

- **Prediction table** (per-label / macro AUROC, macro F1) from
  `evaluation/evaluate.py`, i.e. `evaluation/results/eval_<backbone>.json`. Report
  each number with its **bootstrap 95% CI** (`*_ci` fields), and where you have
  multiple seeds, the **mean +/- std** from `evaluation/aggregate_seeds.py`
  (`aggregate_<backbone>.json`). The `reproducibility` block (seed, git commit)
  belongs in an appendix or footnote so results are regenerable.
- **Localization table** (pointing game, mean IoU, localization accuracy over the
  8 boxed pathologies) from `evaluation/evaluate_localization.py`, i.e.
  `evaluation/results/loc_<backbone>_<method>.json`. This is the table that
  substantiates the explainability claim, so it belongs in the main results
  section next to the prediction table.
- **Ablation table** (classification-only vs. +localization vs. full MIRROR)
  from `evaluation/ablation.py`, i.e. `evaluation/results/ablation_<backbone>.json`.
  This is the baseline comparison the research question names: it shows the
  prediction metrics are unchanged by the added layers (interpretability at no
  predictive cost) while the localization/report capabilities and their latency
  appear only in the relevant rows. Pairs naturally with an ablation/discussion
  section.

Qualitative figures (saliency overlays) come from `demo/run_demo.py` or notebook
`02_pipeline_walkthrough.ipynb`.

> The "Potential Paper Title" and "Expected Deliverables" sections from the
> original project overview were intentionally left out of this scaffold per the
> build brief.
