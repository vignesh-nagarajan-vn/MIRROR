# Revision notes: v9 to v10

A correctness pass over v9. Every fix below responds to a reviewer note; none
changes a measured number, and the paper stays under the 8-page cap. The v9 draft
is kept at
[`pdf-drafts/MIRROR_Paper_Draft_v9.pdf`](pdf-drafts/MIRROR_Paper_Draft_v9.pdf) for
comparison, and this build is
[`pdf-drafts/MIRROR_Paper_Draft_v10.pdf`](pdf-drafts/MIRROR_Paper_Draft_v10.pdf).

Two v9 additions are corrected here: the ResNet baseline was mislabelled
DenseNet-121, and the Guo citation added in v9 was attached to a claim it does not
support. Both are fixed below.

## Claims and numbers corrected

| # | v9 said | v10 says | Justification |
| --- | --- | --- | --- |
| 1 | "The published MedMNIST v2 DenseNet-121 baseline for this dataset is approximately 0.77." | "The published MedMNIST v2 baseline for this dataset, a ResNet-18/50 at 224 px (v2 reports no DenseNet-121), is approximately 0.77." | MedMNIST v2 (Yang et al. 2023) benchmarks ResNet-18 and ResNet-50 at 28 and 224 px plus auto-sklearn, AutoKeras, and Google AutoML Vision. There is no DenseNet-121 baseline; 0.773 is the ResNet-18/50 value at 224 px. The architecture name was wrong, not the number. |
| 2 | Section 5.4: "…a documented pitfall~\cite{guo2017calibration}, not our finding…" | The citation is removed from that sentence (the p(1−p) arithmetic in the same paragraph supports it). Guo et al. 2017 is moved to the ECE definition in Section 4, "Expected Calibration Error~\cite{guo2017calibration}", which it does support. | Guo et al. 2017 is about overconfidence in modern networks and popularised ECE with equal-width bins; it is not about aggregate calibration metrics flattering models under class imbalance. The reference is kept where it is accurate. |
| 6 | Table 3 macro PPV printed as 0.097 while eleven per-label PPVs are "undefined". | Caption: "the Macro PPV averages the eleven undefined labels as zero, and is 0.455 over the three that fire." | `operating_point.macro.ppv` = 0.09740 in `results/chestmnist/eval_chestmnist_densenet121.json`; the eleven undefined labels store `ppv` = 0.0. The three defined PPVs (Effusion 0.599, Infiltration 0.415, Pneumothorax 0.350) mean 0.4545. |
| 7 | Table 3 macro Lift row printed 3.1×, but the caption defines lift = AUPRC / prevalence, which for the macro values is 0.135 / 0.052 = 2.6×. | Caption: "The Macro Lift is the mean of the per-label lifts, not macro AUPRC over macro prevalence (2.6×)." | Mean of the fourteen per-label lifts = 3.108 (→ 3.1×, the value used in the abstract and text). macro_auprc / mean_prevalence = 0.13500 / 0.05222 = 2.585 (→ 2.6×). Both are named. |
| 8 | Table 4 showed a single "ms/study" column with deltas ("141 (+41)", "136 (+0.03)") that did not add to a fixed 101 baseline and referenced different baselines. | Table 4 now has Predict / Localize / Report / Total columns; each row sums (99.5 + 41.4 = 140.9; 100.0 + 36.2 + 0.03 = 136.2). The caption states each row is a separate profiling run whose Predict time carries the run-to-run spread, which is why full MIRROR totals less than localization alone. | `results/chestmnist/ablation_chestmnist_densenet121.json` `profile.latency`: classification prediction 101.18; with-localization prediction 99.48 + localization 41.39 = 140.87; full prediction 100.03 + localization 36.16 + report 0.025 = 136.22. |
| 9 | Table 3 lifts (Fibrosis 1.6, Hernia 5.4) do not equal the ratio of the displayed AUPRC / prevalence (0.025/0.015 = 1.7; 0.011/0.002 = 5.5). | Caption: "Lifts use unrounded AUPRC and prevalence, so a row's Lift can differ from the ratio of its displayed values (Fibrosis 1.6, Hernia 5.4)." | From the JSON: Fibrosis 0.024586 / 0.015417 = 1.595 (→ 1.6); Hernia 0.010763 / 0.002 = 5.381 (→ 5.4). The 1.6 that propagates to the abstract and conclusion is correct from the unrounded values. |
| 10 | "…evaluate on 12,000 held-out test images." | "…evaluate on 12,000 test images, a fixed random subsample (seed 42) of the official 22,433-image test split, which is otherwise held out untouched." | `datasets/scripts/prepare_chestmnist.py`: `--n-test` default 12000 with help "0 = all 22,433"; `_export` draws `rng.permutation(len(imgs))[:n]` with `rng = np.random.default_rng(seed)`, seed 42. The official test split is kept separate from the pooled train+val. |

## Layout

The reading-room UI screenshot (Figure 3) was enlarged to 0.70 of the text width
in v9. The corrections above add roughly a dozen lines (caption clarifications,
the per-stage latency table, the subsample sentence), so at 8 pages the figure
returns to its long-standing 0.56 width. The two result bar charts were left at
full size; shrinking them to 4.6 cm would allow the figure to grow back to about
0.64, but reducing real-data figures to enlarge an illustrative screenshot was not
worth it. No prose was cut to make room beyond tightening the sentences added in
this pass.
