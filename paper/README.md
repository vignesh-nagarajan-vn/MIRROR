# Paper

Scaffold for the MIRROR write-up. Drafts and figures live here; nothing in this
directory is required to run the system.

## Compiling the draft

[`main.tex`](main.tex) is a **single, self-contained** manuscript draft that
compiles as-is in [Overleaf](https://www.overleaf.com/) (or with local
`pdflatex`) to produce a PDF. It needs no external assets: the architecture
figure is drawn inline with TikZ, the bibliography is embedded, and the three
result tables are inlined from the `results/evaluation/*.json` snapshots. Drop it
into a blank Overleaf project and hit *Recompile*.

> The result numbers in `main.tex` are the **illustrative, literature-scale
> placeholders** from `results/` (flagged as such in the manuscript's integrity
> note). Replace them with real values after training on ChestX-ray14 and running
> the `evaluation/` harnesses before making any performance claim.

Suggested structure as the draft grows:

```
paper/
├── main.tex            # manuscript source (self-contained draft — provided)
├── references.bib      # optional: externalize the embedded bibliography
├── figures/            # exported figures (architecture, qualitative overlays)
└── tables/             # generated result tables (from evaluation/results/*.json)
```

Result tables can be generated from the JSON written by the evaluation
harnesses:

- **Prediction table** (per-label / macro AUROC, macro F1) — from
  `evaluation/evaluate.py` → `evaluation/results/eval_<backbone>.json`. Report
  each number with its **bootstrap 95% CI** (`*_ci` fields), and where you have
  multiple seeds, the **mean ± std** from `evaluation/aggregate_seeds.py` →
  `aggregate_<backbone>.json`. The `reproducibility` block (seed, git commit)
  belongs in an appendix or footnote so results are regenerable.
- **Localization table** (pointing game, mean IoU, localization accuracy over the
  8 boxed pathologies) — from `evaluation/evaluate_localization.py` →
  `evaluation/results/loc_<backbone>_<method>.json`. This is the table that
  substantiates the explainability claim, so it belongs in the main results
  section next to the prediction table.
- **Ablation table** (classification-only vs. +localization vs. full MIRROR) —
  from `evaluation/ablation.py` → `evaluation/results/ablation_<backbone>.json`.
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
