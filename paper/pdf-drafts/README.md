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
| `MIRROR_Paper_Draft_v6.pdf` | **Current / submission version.** Finalized for medRxiv (Radiology): grounding claim narrowed to finding-level (descriptive prose is not pixel-verified), explanation metrics framed as a defined protocol with box-level scores as future work, the ablation reframed as a post-hoc correctness check, ChestMNIST framed as a systems demonstration, plus the human-subjects/ethics-approval statement and a Declarations block. Human-toned prose, no em dashes. |

**v6 is the current submission version**, matching [`../main.tex`](../main.tex). To
regenerate it, compile `../main.tex` in Overleaf (upload it with the `../figures/`
folder) or run `pdflatex main` locally. See [`../README.md`](../README.md) for build
details.
