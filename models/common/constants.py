"""Shared constants for MIRROR.

The 14 pathology labels follow the NIH ChestX-ray14 taxonomy. Keeping a single
source of truth here means the classifier head, the report generator, and the
evaluation scripts all agree on label order.
"""

from __future__ import annotations

# NIH ChestX-ray14 disease categories, in the canonical order used by the
# released metadata. Model output index i corresponds to CHESTXRAY14_LABELS[i].
CHESTXRAY14_LABELS: list[str] = [
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
    "Consolidation",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Pleural_Thickening",
    "Hernia",
]

NUM_CHESTXRAY14_CLASSES: int = len(CHESTXRAY14_LABELS)

# ImageNet normalisation — the pretrained torchvision backbones expect this.
IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)

# Default input resolution. 224 keeps us compatible with the standard
# DenseNet/EfficientNet/ViT pretrained weights.
DEFAULT_IMAGE_SIZE: int = 224

# Supported modality keys. The per-modality label sets, descriptions, anatomical
# plane, and report phrasing live in ``models/common/modalities.py``, which the
# pipeline uses to route each study to the right classifier head and vocabulary.
# This tuple is kept only as a lightweight, torch-free advertisement of what is
# available; ``modalities.MODALITY_REGISTRY`` is the source of truth.
MODALITIES: tuple[str, ...] = ("chest_xray", "brain_mri", "head_ct")

# A plain-English description for each label, used to ground the report
# generation prompt so the LLM does not have to recall what each token means.
LABEL_DESCRIPTIONS: dict[str, str] = {
    "Atelectasis": "partial or complete collapse of a lung or lobe",
    "Cardiomegaly": "enlargement of the cardiac silhouette",
    "Effusion": "fluid in the pleural space",
    "Infiltration": "ill-defined opacity suggesting interstitial or alveolar filling",
    "Mass": "a discrete lesion larger than 3 cm",
    "Nodule": "a rounded opacity up to 3 cm",
    "Pneumonia": "airspace opacity consistent with infection",
    "Pneumothorax": "air in the pleural space",
    "Consolidation": "region of lung filled with liquid instead of air",
    "Edema": "fluid accumulation in the interstitial and alveolar spaces",
    "Emphysema": "lung hyperinflation with destruction of alveolar walls",
    "Fibrosis": "scarring and architectural distortion of lung tissue",
    "Pleural_Thickening": "thickening of the pleural lining",
    "Hernia": "protrusion of abdominal contents into the thorax",
}
