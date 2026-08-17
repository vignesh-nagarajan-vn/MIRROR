# IEEE workshop variants

Three IEEEtran conference-format versions of the MIRROR paper, one per target IEEE
workshop. Each is a **reframing** of the SSRN/arXiv preprint
([`../main.tex`](../main.tex), v10), not a new study: every figure, table, and
measured number is copied from the source paper, which traces to committed JSON
under [`../../results/`](../../results/) and to
[`../calibration_baseline.py`](../calibration_baseline.py). No result changed. Only
the framing (title, abstract, introduction, one workshop-specific section, the
discussion emphasis, and the conclusion) differs between them.

The source `main.tex` is a single-column-title `article`-class draft capped at 8
pages for SSRN/arXiv. These three are two-column IEEEtran, so they read differently
even where the words are identical.

## The three variants

| File | Target workshop | Pages | Customized emphasis |
| --- | --- | --- | --- |
| [`mirror_trustworthy_ai_pipelines.tex`](mirror_trustworthy_ai_pipelines.tex) | Building Trustworthy AI Pipelines for Big Data: Verification, Provenance, and Reproducibility | 9 | Trust decomposed into the workshop's three named pillars. Dedicated section maps the grounding unit test and the post-hoc invariance regression test to *verification*, the number-to-JSON traceability and `calibration_baseline.py` to *provenance*, and the seeded, torch-free, versioned runs to *reproducibility*. |
| [`mirror_multimodal_medical_data.tex`](mirror_multimodal_medical_data.tex) | The 6th International Workshop on Multi-Modal Medical Data Analysis | 9 | Multi-modality two ways. Across imaging modalities (chest X-ray, brain MRI, head CT) via the registry and DICOM-tag routing, and across representations (image to structured evidence to text) as a grounded cross-modal translation. Dedicated "Multi-Modal Analysis by Construction" section. |
| [`mirror_bigdata_analytics_medical_imaging.tex`](mirror_bigdata_analytics_medical_imaging.tex) | 3rd Workshop on Big Data Analytics for Medical Imaging | 9 | Analytics over large, heterogeneous corpora: DICOM ingestion as data engineering, the registry as a way to scale across modalities, post-hoc stages as a throughput cost, and the methodological warning that aggregate metrics flatter models on large imbalanced data unless read against no-skill floors. |

Every variant lands within the requested 8 to 10 page range.

## Target venues and registration cost

All three target workshops are half-day workshops of the **IEEE International
Conference on Big Data (IEEE BigData)**, whose 2026 edition is in Phoenix, Arizona,
December 14 to 17, 2026. They are the next editions of workshops that ran at BigData
2025: Big Data Analytics for Medical Imaging was the 2nd there, Multi-Modal Medical
Data Analysis was the 5th, and the Trustworthy AI Pipelines workshop is not in the
2025 list, so it appears to be a new (first) edition.

Because all three sit under the same conference, **registration cost is the same for
all three**. IEEE BigData requires at least one author of each accepted paper,
workshops included, to pay the full author registration, and there is no cheaper
workshop-only author rate. The 2026 fees are not published yet; the most recent
published rates, from BigData 2024, were:

| Category | Early-bird | Regular |
| --- | --- | --- |
| IEEE member | $850 | $950 |
| Non-member | $1,020 | $1,140 |
| IEEE student member | $595 | $695 |
| Student non-member | $715 | $835 |

Expect the 2026 numbers near these, likely a little higher. The cost is the author
registration, not the choice of workshop; travel to Phoenix is separate and not
counted here.

**Submission recommendation, on acceptance odds alone:** the Trustworthy AI
Pipelines workshop. The paper's contributions map directly onto its named themes of
verification, provenance, and reproducibility, and its deliberately modest empirical
result reads there as honest reporting rather than as a weak result, which is how it
would read at a results-focused or scale-focused venue.

## What every variant keeps identical (the facts)

- The three-layer pipeline: classification, Grad-CAM evidence localization, and a
  report layer whose language stage receives structured evidence and never pixels.
- The finding-level vs detail-level grounding distinction, and the worked example
  of an ungrounded cardiothoracic ratio in a real generated report.
- The ChestMNIST result: macro AUROC 0.729, per-label panel, 11 of 14 labels silent
  at threshold 0.5, AUPRC lift 1.6x to 6.8x, Brier 0.045 against a 0.047
  constant-prevalence floor, ECE 0.018, bootstrap B=1000.
- The synthetic harness control, the post-hoc invariance regression test (max
  probability delta 0.000), and the per-stage latency profile.
- The honest scope: **only chest X-ray is trained and benchmarked**; brain MRI and
  head CT are registered, routed, and tested, with no trained checkpoint, so no
  predictive claim is made on them.

## IEEE compliance

Built against the attached IEEEtran class guide. Choices made for compliance:

- `\documentclass[conference]{IEEEtran}`: two-column, 10pt, US letter, the standard
  conference layout.
- **No font-override package.** Per Appendix D of the guide ("allow IEEEtran to
  manage the fonts"), the source paper's `newtxtext`/`newtxmath` are dropped and the
  class default is used. Compiles with zero overfull boxes.
- `cite.sty` for IEEE-style bracketed, sorted, compressed citations.
- Figure captions below figures, table captions above tables, "Fig." abbreviation,
  centered floats. Two-column-wide floats via `figure*`/`table*`.
- Bibliography is an embedded `thebibliography` (24 sources), so there is no `bibtex`
  step and the file is self-contained.
- The three UI screenshots are pulled from [`../figures/`](../figures/) and guarded
  by `\IfFileExists`, so each file still compiles (with a placeholder box) if the
  images are absent.

## Build

Compile from this directory so the `../figures/` paths resolve. Two pdfLaTeX passes
resolve the cross-references:

```bash
cd paper/ieee-workshops
pdflatex mirror_trustworthy_ai_pipelines
pdflatex mirror_trustworthy_ai_pipelines
```

Repeat for the other two file stems. Each produces a 9-page PDF. Build
intermediates (`.aux`, `.log`, `.out`) are git-ignored; the `.tex` sources and the
compiled `.pdf` are committed.

## Relationship to the preprint

The canonical version of this work is the SSRN preprint (abstract 7245078,
CC BY-NC-ND) and the matching arXiv cs.CV submission. These three files are
venue-specific presentations of that same work and introduce no new claims. If a
number in the preprint is corrected, correct it here too; the single source of
truth for every result remains the committed JSON and `calibration_baseline.py`.
