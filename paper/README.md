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

Result tables can be generated from the JSON written by `evaluation/evaluate.py`.
Qualitative figures (saliency overlays) come from `demo/run_demo.py` or notebook
`02_pipeline_walkthrough.ipynb`.

> The "Potential Paper Title" and "Expected Deliverables" sections from the
> original project overview were intentionally left out of this scaffold per the
> build brief.
