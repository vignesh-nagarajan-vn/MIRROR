"""Tests for the modality registry (``models/common/modalities.py``).

Torch-free: the registry, its resolution rules, and the location vocabulary have
no ML dependencies, so these run anywhere. They pin the behaviour the rest of the
pipeline relies on: stable label sets, forgiving free-text/DICOM resolution, and
plane-appropriate location descriptions.

Run:  pytest tests/test_modalities.py
"""

from __future__ import annotations

from models.common.constants import CHESTXRAY14_LABELS
from models.common.modalities import (
    DEFAULT_MODALITY,
    MODALITY_REGISTRY,
    list_modalities,
    modality_from_dicom_tag,
    resolve_modality,
)
from models.explainability.explainer import describe_location


def test_registry_has_three_modalities_with_distinct_labels():
    keys = set(MODALITY_REGISTRY)
    assert keys == {"chest_xray", "brain_mri", "head_ct"}
    # Every modality has a non-empty, de-duplicated label set.
    for spec in MODALITY_REGISTRY.values():
        assert spec.num_labels == len(set(spec.labels)) > 0
        # Each label is glossed for the report prompt.
        assert set(spec.labels) <= set(spec.label_descriptions)


def test_chest_labels_match_single_source_of_truth():
    # The registry must reuse constants.py, not fork the chest label list.
    assert MODALITY_REGISTRY["chest_xray"].labels == tuple(CHESTXRAY14_LABELS)


def test_resolve_by_display_alias_and_key():
    assert resolve_modality("Brain MRI").key == "brain_mri"
    assert resolve_modality("brain mri").key == "brain_mri"
    assert resolve_modality("CT").key == "head_ct"
    assert resolve_modality("head_ct").key == "head_ct"
    assert resolve_modality("chest X-ray").key == "chest_xray"


def test_resolve_unknown_and_auto_fall_back_to_default():
    for value in (None, "", "auto", "totally unknown modality"):
        assert resolve_modality(value).key == DEFAULT_MODALITY


def test_resolve_bare_dicom_value():
    assert resolve_modality("MR").key == "brain_mri"
    assert resolve_modality("CT").key == "head_ct"
    assert resolve_modality("CR").key == "chest_xray"


def test_modality_from_dicom_tag():
    assert modality_from_dicom_tag("MR") == "brain_mri"
    assert modality_from_dicom_tag("ct") == "head_ct"  # case-insensitive
    assert modality_from_dicom_tag("DX") == "chest_xray"
    assert modality_from_dicom_tag("US") is None  # ultrasound: not supported
    assert modality_from_dicom_tag(None) is None


def test_describe_location_frontal_uses_lung_zones():
    # Radiographic convention: patient-right is viewer-left (small x -> "right").
    assert describe_location((0.1, 0.1), "frontal") == "the upper right zone"
    assert describe_location((0.5, 0.5), "frontal") == "the mid central zone"
    assert describe_location((0.9, 0.9), "frontal") == "the lower left zone"


def test_describe_location_axial_uses_brain_regions():
    assert describe_location((0.1, 0.1), "axial") == "the right frontal region"
    assert describe_location((0.5, 0.5), "axial") == "the parietal midline region"
    assert describe_location((0.9, 0.9), "axial") == "the left occipital region"


def test_describe_location_defaults_to_frontal():
    # Backward compatibility with the original chest-only signature.
    assert describe_location((0.5, 0.5)) == "the mid central zone"


def test_list_modalities_is_json_shaped():
    listed = list_modalities()
    assert [m["key"] for m in listed] == ["chest_xray", "brain_mri", "head_ct"]
    for m in listed:
        assert set(m) == {"key", "display_name", "labels", "count", "plane"}
        assert m["count"] == len(m["labels"])
