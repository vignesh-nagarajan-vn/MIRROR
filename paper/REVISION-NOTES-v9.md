# Revision notes: v8 to v9

An accuracy pass on the SSRN preprint ([abstract
7245078](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7245078)). The
engineering did not change. The framing did, because v8 claimed more than its
evidence supported in eight specific places.

Every number in v9 traces to a JSON committed under [`../results/`](../results/)
or to [`calibration_baseline.py`](calibration_baseline.py). The v8 draft is kept
at [`pdf-drafts/MIRROR_Paper_Draft_v8.pdf`](pdf-drafts/MIRROR_Paper_Draft_v8.pdf)
for comparison.

## Claims changed

| # | v8 said | v9 says | Justification |
| --- | --- | --- | --- |
| 1 | "Interpretability at no predictive cost" in the abstract, contributions, a section heading, the discussion, and the conclusion. | The phrase is gone. The invariance check appears once, in Section 5.5, labelled a regression test. | `ablation.py` runs layers 2 and 3 after the forward pass; they touch no weights and no logits, so `max_prob_delta = 0.0` holds by construction. A tautology is not a finding. |
| 2 | Table 3 showed macro AUROC 0.729 in three rows as if three results. | Table 4 has no AUROC column. One line under the rule states macro AUROC = 0.729 applies to all three rows, one measurement. The caption says so in bold. | `evaluation/ablation.py:153` copies a single `macro_auroc` into every row under the comment "same across conditions by construction". |
| 3a | Fig. 2 captioned "a real session from the live demo", implying the benchmarked DenseNet-121. | Both screenshot captions state the hosted vision-LLM engine produced them and that no number in them is a measured result. Section 3.5 and Table 2 state the grounding guarantee does not hold on the hosted path. | `frontend/app/api/analyze/route.ts:251-277` sends the image to `claude-haiku-4-5` under a forced `record_analysis` tool call returning all 14 probabilities, a bbox per finding, and the report. `docs/deployment-showcase.md:63-67` confirms these are not DenseNet or Grad-CAM outputs. |
| 3b | The fabricated clinical detail in the Fig. 3b report was unmentioned. | Section 7 quotes "cardiothoracic ratio below 0.5" and "costophrenic angles are clear" as a worked example of ungrounded detail-level prose, and argues fluency makes it more dangerous because it borrows credibility from the adjacent verified finding. | `docs/deployment-showcase.md:130-132`. No component of the pipeline computes a cardiothoracic ratio or localizes a costophrenic angle. |
| 4 | "At the fixed 0.5 threshold it is deliberately conservative (macro specificity 0.997, sensitivity 0.019)." | Section 5.2 plus Table 3: 11 of 14 labels emit no positive prediction at all; the three that fire are named; macro F1 0.031; "at this threshold the classifier is not usable as a detector", a property of the training budget, not a design choice. | `results/chestmnist/eval_chestmnist_densenet121.json` `operating_point.per_label`: 11 labels have `sensitivity` exactly 0.0 and `specificity` exactly 1.0. Only Effusion (0.143), Infiltration (0.110), Pneumothorax (0.012) are nonzero. |
| 5 | Brier 0.045 / ECE 0.018 presented as "well calibrated in aggregate". | New Section 5.3: a constant-prevalence predictor scores macro Brier 0.0472; the model scores 0.0453; the whole advantage is 0.0019, about 4%. Named as a finding and moved into the abstract. | Computed by [`calibration_baseline.py`](calibration_baseline.py) from `support_pos`/`support_neg` in the same JSON. For prevalence p, a constant predictor scores exactly p(1-p). |
| 6a | Learning rate 1e-4. | 3e-4. | `configs/chestmnist.yaml` `train.lr: 0.0003`. The 1e-4 belongs to `configs/default.yaml`, which was not the config used. |
| 6b | 224px and 64px both mentioned, never connected. | "Source images are 64x64 and are upsampled to the backbone's 224x224 input: there is no genuine 224-pixel information anywhere in this experiment." | `prepare_chestmnist.py --size 64`; `configs/chestmnist.yaml` `data.image_size: 224`. |
| 6c | "within reach of the published MedMNIST baseline (AUROC ~0.77)". | "ours is 0.04 AUROC below it", with the compute difference stated as a fact and no inference drawn. | 0.729 < 0.77. The baseline is a literature value cited to MedMNIST v2, not a repo measurement. |
| 6d | "7,200 of the 78,468 training images". | "We sample 8,000 studies from the pooled official train and validation splits and hold out 10%, giving 7,200 training and 800 validation images." | `prepare_chestmnist.py` pools train+val then samples `--n-train-val 8000`; `models/classification/train.py:66` holds out 10%. The 78,468 figure is the train split alone, so the original phrasing was wrong. |
| 7 | Research question asked whether the system "improves interpretability without degrading predictive performance". | "Can a radiology pipeline be built so that its language layer is structurally incapable of asserting a finding the classifier did not detect, and what does that constraint cost?" Section 1 then states that explanation quality is not measured and names what would measure it (running the implemented pointing-game/IoU protocol on a full-resolution checkpoint, and a reader study). | The second clause of the v8 question is true by construction (see #1). The first was never measured: no localization harness run exists and no reader study was conducted. |
| 8 | Adebayo and Rudin cited, neither engaged. | A Discussion paragraph concedes MIRROR is the post-hoc architecture Rudin argues against, declines to claim saliency faithfulness, notes the Adebayo sanity checks were not run, and defends only auditability of the finding set. Doshi-Velez and Kim cut as decorative. | Neither the sanity checks nor a Rudin response existed in v8. |
| 9 | Title blurb: "predicts findings across three imaging modalities". Contribution 3 claimed "a modality-agnostic realization across chest X-ray, brain MRI, and head CT". | Scope line: "three modality taxonomies are registered and routed, one of which, chest X-ray, is trained and benchmarked here." Table 1 caption defines what registration means. Section 3.4 and the limitations state MIRROR makes no predictive claim on MRI or CT. | No brain-MRI or head-CT checkpoint exists in the repo. `tests/test_modalities.py`, `test_pipeline_routing.py`, and `test_report_modality.py` exercise routing and vocabulary, which is what is now claimed. |

## Numbers corrected against the repo

Two figures in v8 did not survive recomputation. Both were corrected in the paper
and in the repo docs that carried them.

| Value | v8 / repo docs | Correct | Source |
| --- | --- | --- | --- |
| Labels with sensitivity 0.0 | (not reported) | **11 of 14** | Eleven labels in `operating_point.per_label` have sensitivity exactly 0.0. |
| Synthetic no-signal mean AUROC | 0.557 | **0.533** | Mean of the seven no-signal per-label AUROCs in `eval_synthetic_densenet121.json` is 0.53306. Cross-check: (0.91689 + 0.53306) / 2 = 0.72497, which is the JSON's macro exactly. The 0.557 originated in `results/synthetic_validation/README.md` and was copied into the paper; both are now fixed. |

## Kept and given more room

- The finding-level vs detail-level grounding distinction, promoted from a
  paragraph in Section 7 to the lead of the Discussion and to the abstract.
- The DICOM ingest, now with the reason it matters (modality LUT, VOI/window LUT,
  MONOCHROME1 inversion, and what silently goes wrong when they are skipped).
- The synthetic harness control, 0.917 signal vs 0.533 no-signal, with a note that
  the no-signal mean sits slightly above 0.5 as a finite test split should.
- Reproducibility discipline: seeds, git commit, and library versions stamped into
  every results JSON, torch-free unit tests, and `results/README.md` distinguishing
  real from illustrative outputs.
- The bootstrap detail: one resample drives all metrics, so the intervals are
  mutually consistent.

## Cut for the page budget

Literature review compressed by about a third with the dataset and explanation
paragraphs merged; the CheXpert, CAM, and Doshi-Velez citations dropped as
decorative (21 references down to 16); contributions trimmed from 5 to 4; the old
5.2 subsection collapsed; Discussion de-duplicated against Results and Conclusion;
the per-label AUROC table cut as fully redundant with the per-label AUROC figure;
Section 3.5 and Table 2 prose tightened.

## Verification

- Compiles with `pdflatex` in a single pass, no bibtex. Zero undefined references
  or citations.
- 8 pages exactly (`pdfinfo`).
- No string from `results/evaluation/*.json` (the hand-written placeholders)
  appears in the PDF; the placeholder localization numbers 0.6311, 0.2218, 0.6748
  and the 984-box count were all checked absent.
- The banned v8 phrases were checked absent: "at no predictive cost", "within
  reach", "deliberately conservative", "1e-4" as the learning rate, and any claim
  of prediction across three modalities.

## One number that is not repo-traceable

The MedMNIST v2 DenseNet-121 baseline of approximately 0.77 AUROC is a literature
value cited to Yang et al. (2023), not a measurement from this repository. It is
used only to state that our 0.729 is below it. Every other number in the PDF comes
from `results/chestmnist/`, `results/synthetic_validation/`, or
`calibration_baseline.py`, except dataset-size facts cited to the NIH ChestX-ray14
release.
