# Paper

Scaffold for the MIRROR write-up. Drafts and figures live here; nothing in this
directory is required to run the system.

Suggested structure:

```
paper/
├── main.tex            # manuscript source
├── references.bib      # citations (Grad-CAM, Score-CAM, ChestX-ray14, MIMIC-CXR…)
├── figures/            # exported figures (architecture, qualitative overlays)
└── tables/             # generated result tables (from evaluation/results/*.json)
```

Result tables can be generated from the JSON written by the two evaluation
harnesses:

- **Prediction table** (per-label / macro AUROC, macro F1) — from
  `evaluation/evaluate.py` → `evaluation/results/eval_<backbone>.json`.
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
