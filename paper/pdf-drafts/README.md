# PDF drafts

Compiled PDF snapshots of the MIRROR paper draft, kept for quick reference and
version comparison. The source of truth is [`../main.tex`](../main.tex); these
PDFs are rendered outputs and are not regenerated automatically.

Versions are listed oldest to newest; the last row is the current state of
[`../main.tex`](../main.tex).

| File | Notes |
| --- | --- |
| `MIRROR_Paper_Draft_v1.pdf` | Initial compiled draft (chest-radiograph only, illustrative placeholder results). |
| `MIRROR_Paper_Draft_v2.pdf` | Revised chest-only draft. |
| `MIRROR_Paper_Draft_v3.pdf` | Multi-modality (chest X-ray / brain MRI / head CT) and the expanded clinical-metric panel, with the measured ChestMNIST results folded in; still carried red placeholders for the full-resolution NIH numbers. |
| `MIRROR_Paper_Draft_v4.pdf` | Later snapshot of the same multi-modality / measured-ChestMNIST draft. |
| `MIRROR_Paper_Draft_v5.pdf` | All-measured paper: every red placeholder removed, Results rebuilt from real numbers only (ChestMNIST per-label AUROC, measured ablation, synthetic sanity check), shortened toward 8 pages. |
| `MIRROR_Paper_Draft_v6.pdf` | First medRxiv (Radiology) submission version: grounding claim narrowed to finding-level (descriptive prose is not pixel-verified), explanation metrics framed as a defined protocol with box-level scores as future work, the ablation reframed as a post-hoc correctness check, ChestMNIST framed as a systems demonstration, plus the human-subjects/ethics-approval statement and a Declarations block. Human-toned prose, no em dashes. |
| `MIRROR_Paper_Draft_v7.pdf` | Resubmission version for medRxiv, prepared after screening asked for author details and institutional affiliation clarified. The title page carries a single centered corresponding-author block (Vignesh Nagarajan, Texas A&M University), the mentor moves to a bolded Acknowledgments entry, the internship affiliation moves off the title page into Acknowledgments, and the body is condensed to fit the 8-page maximum. medRxiv later declined the resubmission, judging the work a technical development rather than biomedical research. |
| `MIRROR_Paper_Draft_v8.pdf` | Submitted to SSRN on August 6, 2026 ([abstract 7245078](https://ssrn.com/abstract=7245078), CC BY-NC-ND). Restores the v6 two-author title block. The research mentor sits beside the corresponding author with his qualifiers (Research Mentor; Applied AI Researcher, Capital One; PhD in Computer Science, IIT Hyderabad), and the Global Indian Scientists & Technocrats (GIST) 2026 Summer Research Internship affiliation runs on one centered line beneath both author blocks. Body unchanged from v7, 8 pages. |
| `MIRROR_Paper_Draft_v9.pdf` | **Current, replacing v8 on SSRN.** Accuracy revision plus expansion, no new experiments. The post-hoc invariance check is demoted from a headline claim to a regression test ("interpretability at no predictive cost" is gone, since a delta of zero is true by construction), and the ablation table no longer prints one AUROC measurement as three results. Both screenshot figures state that the hosted vision-LLM engine produced them, and the paper says plainly that the grounding guarantee holds for the local PyTorch stack only. Newly reported: the operating point (11 of 14 labels emit no positive prediction at threshold 0.5), a calibration baseline (a constant-prevalence predictor scores Brier 0.047 against the model's 0.045), and AUPRC as lift over prevalence (every label ranks 1.6x to 6.8x above chance, so discrimination is real where decisions are absent). Adds the full 14-row clinical panel, the Layer 3 evidence-payload figure, and the synthetic-control chart. Learning rate, resolution, and split counts corrected against the configs. Retypeset on `newtxtext`/`newtxmath` with widow, orphan, and double-float controls: no stranded words, no overfull boxes, no half-empty float pages. Abstract rewritten problem-first at 260 words. Title block, authors, acknowledgments, and declarations unchanged. 8 pages. |

**v9 is the current version** ([SSRN abstract
7245078](https://ssrn.com/abstract=7245078)), matching [`../main.tex`](../main.tex)
(8 pages, with all three UI screenshots embedded). To regenerate it, compile
`../main.tex` in Overleaf (upload it with the `../figures/` folder) or run
`pdflatex main` locally. See [`../README.md`](../README.md) for build details and
[`../REVISION-NOTES-v9.md`](../REVISION-NOTES-v9.md) for what changed and why.
