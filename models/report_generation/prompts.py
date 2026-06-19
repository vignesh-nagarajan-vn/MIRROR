"""Prompt construction for the clinical reasoning layer.

The language model never sees the image directly in this layer — it reasons over
the *structured evidence* produced upstream: the classifier's per-label
probabilities and the explainability module's localisation of each positive
finding. This keeps the LLM grounded and makes the report auditable: every
sentence traces back to a probability and a heatmap region.
"""

from __future__ import annotations

from ..common.constants import LABEL_DESCRIPTIONS

SYSTEM_PROMPT = (
    "You are a radiology reporting assistant. You write structured, clinician-"
    "style draft reports from structured model evidence. You never invent "
    "findings that are not in the evidence. You always include an explicit "
    "statement that the report is AI-generated and requires review by a "
    "licensed radiologist. You are concise and use standard radiology section "
    "headings: FINDINGS and IMPRESSION."
)


def build_evidence_block(findings: list[dict]) -> str:
    """Render the structured evidence the model must reason over.

    Each finding dict has: label, probability, location (str), present (bool).
    """
    lines = []
    for f in findings:
        desc = LABEL_DESCRIPTIONS.get(f["label"], "")
        status = "PRESENT" if f["present"] else "below threshold"
        loc = f.get("location", "n/a")
        lines.append(
            f"- {f['label']} ({desc}): probability={f['probability']:.2f}, "
            f"status={status}, localised to {loc}"
        )
    return "\n".join(lines)


def build_user_prompt(
    findings: list[dict],
    modality: str = "chest X-ray",
    indication: str | None = None,
) -> str:
    """Assemble the full user-turn prompt for the report model."""
    evidence = build_evidence_block(findings)
    indication_line = (
        f"Clinical indication: {indication}\n" if indication else ""
    )
    return (
        f"Modality: {modality}\n"
        f"{indication_line}\n"
        "Structured model evidence (probabilities from an image classifier, "
        "locations from a Grad-CAM saliency map):\n"
        f"{evidence}\n\n"
        "Write a draft report with two sections:\n"
        "FINDINGS: describe each present finding, referencing its location and "
        "the model's confidence in qualitative terms (e.g. 'high confidence'). "
        "Note pertinent negatives for the major findings that were below "
        "threshold.\n"
        "IMPRESSION: a brief prioritised summary.\n\n"
        "End with: 'AI-GENERATED DRAFT — requires verification by a licensed "
        "radiologist.'"
    )
