"""Modality registry for MIRROR.

MIRROR started as a chest-radiograph system, but the pipeline (predict -> localize
-> report) is modality-agnostic: only the *label set*, the anatomical vocabulary
used to describe a location, and a little report phrasing change between a chest
X-ray, a brain MRI, and a head CT. This module is the single source of truth for
that per-modality configuration, the same role ``constants.py`` plays for the 14
ChestX-ray14 labels (which are reused here so there is still exactly one chest
label list).

Each modality is a :class:`ModalitySpec`:

* ``labels`` / ``label_descriptions`` — the finding taxonomy the classifier head
  predicts and the plain-English gloss that grounds the report prompt.
* ``plane`` — ``"frontal"`` (a chest film) or ``"axial"`` (a cross-sectional
  MRI/CT slice). This selects how a saliency centroid is turned into words
  (lung zones vs. brain regions) in ``explainability.explainer.describe_location``.
* ``aliases`` / ``dicom_modalities`` — how a free-text modality string from the
  UI or a DICOM ``Modality`` tag resolves to this spec.
* ``normal_impression`` / ``report_guidance`` — modality-appropriate phrasing so
  a normal brain study is not described as "no acute cardiopulmonary abnormality".

The taxonomies are deliberately grounded in public benchmarks so the plumbing is
clinically sensible even though no trained weights ship with the repo:

* chest X-ray  -> NIH ChestX-ray14 (14 findings).
* brain MRI    -> the common tumour classes of the Brain Tumor MRI Dataset
  (glioma / meningioma / pituitary) plus routine neuro findings.
* head CT      -> the RSNA Intracranial Hemorrhage taxonomy (the five bleed
  subtypes) plus routine acute head-CT findings.

Nothing here reasons over pixels, so the whole module is torch-free and safe to
import anywhere (tests, the Vercel-mirroring code generator, label lookups).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from .constants import CHESTXRAY14_LABELS, LABEL_DESCRIPTIONS


@dataclass(frozen=True)
class ModalitySpec:
    """Everything the pipeline needs to run one imaging modality."""

    key: str                                   # machine key, e.g. "brain_mri"
    display_name: str                          # human label, e.g. "Brain MRI"
    labels: tuple[str, ...]                    # finding taxonomy (output order = index)
    label_descriptions: Mapping[str, str]      # label -> plain-English gloss
    plane: str                                 # "frontal" | "axial"
    normal_impression: str                     # "no acute finding" phrasing
    report_guidance: str                       # extra hint for the report prompt
    aliases: tuple[str, ...] = ()              # free-text names that resolve here
    dicom_modalities: tuple[str, ...] = ()     # DICOM (0008,0060) Modality values

    @property
    def num_labels(self) -> int:
        return len(self.labels)


# --------------------------------------------------------------------------- #
# Chest X-ray (NIH ChestX-ray14) — reuses the constants.py source of truth.
# --------------------------------------------------------------------------- #
CHEST_XRAY = ModalitySpec(
    key="chest_xray",
    display_name="Chest X-ray",
    labels=tuple(CHESTXRAY14_LABELS),
    label_descriptions=dict(LABEL_DESCRIPTIONS),
    plane="frontal",
    normal_impression="No acute cardiopulmonary abnormality detected.",
    report_guidance=(
        "This is a frontal chest radiograph. Describe findings using standard "
        "thoracic anatomy (lung zones, cardiomediastinal silhouette, pleura, "
        "costophrenic angles)."
    ),
    aliases=("chest x-ray", "chest xray", "chest", "cxr", "radiograph", "xray", "x-ray"),
    dicom_modalities=("CR", "DX", "DR"),
)


# --------------------------------------------------------------------------- #
# Brain MRI — tumour classes (Brain Tumor MRI Dataset) + routine neuro findings.
# --------------------------------------------------------------------------- #
BRAIN_MRI_LABELS: tuple[str, ...] = (
    "Glioma",
    "Meningioma",
    "Pituitary_Adenoma",
    "Metastasis",
    "Acute_Infarct",
    "Hemorrhage",
    "Edema",
    "Mass_Effect",
    "Hydrocephalus",
    "White_Matter_Hyperintensity",
    "Atrophy",
)

BRAIN_MRI_DESCRIPTIONS: dict[str, str] = {
    "Glioma": "an intra-axial tumour arising from glial tissue",
    "Meningioma": "an extra-axial dural-based tumour",
    "Pituitary_Adenoma": "a sellar/suprasellar tumour of the pituitary gland",
    "Metastasis": "one or more secondary intracranial tumour deposits",
    "Acute_Infarct": "acute ischaemic injury with restricted diffusion",
    "Hemorrhage": "intracranial blood products",
    "Edema": "vasogenic or cytotoxic parenchymal swelling",
    "Mass_Effect": "displacement of adjacent structures by a lesion",
    "Hydrocephalus": "enlargement of the ventricular system",
    "White_Matter_Hyperintensity": "T2/FLAIR-hyperintense white-matter signal",
    "Atrophy": "volume loss with sulcal and ventricular prominence",
}

BRAIN_MRI = ModalitySpec(
    key="brain_mri",
    display_name="Brain MRI",
    labels=BRAIN_MRI_LABELS,
    label_descriptions=BRAIN_MRI_DESCRIPTIONS,
    plane="axial",
    normal_impression="No acute intracranial abnormality detected.",
    report_guidance=(
        "This is an axial brain MRI slice. Describe findings using neuroanatomy "
        "(frontal/parietal/temporal/occipital lobes, cerebellum, ventricles, "
        "midline) and note laterality. Do not use thoracic terminology."
    ),
    aliases=("brain mri", "mri brain", "mri", "brain", "head mri", "neuro mri"),
    dicom_modalities=("MR",),
)


# --------------------------------------------------------------------------- #
# Head CT — RSNA Intracranial Hemorrhage subtypes + routine acute head-CT findings.
# --------------------------------------------------------------------------- #
HEAD_CT_LABELS: tuple[str, ...] = (
    "Intracranial_Hemorrhage",
    "Intraparenchymal",
    "Intraventricular",
    "Subarachnoid",
    "Subdural",
    "Epidural",
    "Acute_Infarct",
    "Mass_Effect",
    "Midline_Shift",
    "Hydrocephalus",
    "Skull_Fracture",
)

HEAD_CT_DESCRIPTIONS: dict[str, str] = {
    "Intracranial_Hemorrhage": "any acute intracranial blood (parent finding)",
    "Intraparenchymal": "haemorrhage within the brain parenchyma",
    "Intraventricular": "haemorrhage within the ventricular system",
    "Subarachnoid": "haemorrhage in the subarachnoid space",
    "Subdural": "a crescentic collection between dura and arachnoid",
    "Epidural": "a lenticular collection between skull and dura",
    "Acute_Infarct": "an acute ischaemic territory of low attenuation",
    "Mass_Effect": "effacement or displacement of adjacent structures",
    "Midline_Shift": "displacement of midline structures across the falx",
    "Hydrocephalus": "enlargement of the ventricular system",
    "Skull_Fracture": "a lucent cortical break in the calvarium or skull base",
}

HEAD_CT = ModalitySpec(
    key="head_ct",
    display_name="Head CT",
    labels=HEAD_CT_LABELS,
    label_descriptions=HEAD_CT_DESCRIPTIONS,
    plane="axial",
    normal_impression="No acute intracranial abnormality detected.",
    report_guidance=(
        "This is an axial non-contrast head CT slice. Describe findings using "
        "neuroanatomy and standard haemorrhage subtypes; note laterality and any "
        "midline shift. Do not use thoracic terminology."
    ),
    aliases=("ct", "head ct", "brain ct", "ct head", "ct brain", "cct", "computed tomography"),
    dicom_modalities=("CT",),
)


# --------------------------------------------------------------------------- #
# Registry + resolution.
# --------------------------------------------------------------------------- #
DEFAULT_MODALITY: str = "chest_xray"

MODALITY_REGISTRY: dict[str, ModalitySpec] = {
    CHEST_XRAY.key: CHEST_XRAY,
    BRAIN_MRI.key: BRAIN_MRI,
    HEAD_CT.key: HEAD_CT,
}

# Every DICOM (0008,0060) Modality value we know how to route, built from the
# registry so it stays in sync automatically.
_DICOM_MODALITY_MAP: dict[str, str] = {
    dm.upper(): spec.key
    for spec in MODALITY_REGISTRY.values()
    for dm in spec.dicom_modalities
}

# Alias / display-name / key lookup, normalised to lowercase.
_ALIAS_MAP: dict[str, str] = {}
for _spec in MODALITY_REGISTRY.values():
    _ALIAS_MAP[_spec.key.lower()] = _spec.key
    _ALIAS_MAP[_spec.display_name.lower()] = _spec.key
    for _alias in _spec.aliases:
        _ALIAS_MAP.setdefault(_alias.lower(), _spec.key)


def resolve_modality(name: str | None) -> ModalitySpec:
    """Resolve a free-text modality name (or ``None``/``"auto"``) to a spec.

    Accepts the machine key, the display name, any registered alias, or a DICOM
    Modality value, case-insensitively. Unknown or empty values fall back to the
    default modality (chest X-ray) so callers never crash on an unexpected string.
    """
    if not name:
        return MODALITY_REGISTRY[DEFAULT_MODALITY]
    norm = str(name).strip().lower()
    if norm in ("", "auto"):
        return MODALITY_REGISTRY[DEFAULT_MODALITY]
    if norm in _ALIAS_MAP:
        return MODALITY_REGISTRY[_ALIAS_MAP[norm]]
    # A bare DICOM Modality value (e.g. "MR", "CT").
    if norm.upper() in _DICOM_MODALITY_MAP:
        return MODALITY_REGISTRY[_DICOM_MODALITY_MAP[norm.upper()]]
    return MODALITY_REGISTRY[DEFAULT_MODALITY]


def modality_from_dicom_tag(dicom_modality: str | None) -> str | None:
    """Map a DICOM (0008,0060) Modality value to a registry key, or ``None``."""
    if not dicom_modality:
        return None
    return _DICOM_MODALITY_MAP.get(str(dicom_modality).strip().upper())


def list_modalities() -> list[dict]:
    """A JSON-serialisable summary of every modality, for the API/UI."""
    return [
        {
            "key": s.key,
            "display_name": s.display_name,
            "labels": list(s.labels),
            "count": s.num_labels,
            "plane": s.plane,
        }
        for s in MODALITY_REGISTRY.values()
    ]
