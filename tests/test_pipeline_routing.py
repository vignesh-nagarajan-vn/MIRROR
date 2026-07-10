"""Tests for modality routing in the pipeline (``models/pipeline.py``).

The real pipeline needs torch (the classifier + Grad-CAM), but the *routing*
logic - pick the right label set per modality, build one engine per modality and
cache it, describe locations in the right anatomical plane, and keep predictions
invariant to the post-hoc layers - is pure orchestration. These tests stub the
classifier and explainer with lightweight fakes so that orchestration is
exercised without torch, ImageNet weights, or a network.

Run:  pytest tests/test_pipeline_routing.py
"""

from __future__ import annotations

import pytest

import models.pipeline as pipeline_mod
from models.pipeline import MirrorPipeline
from models.common.config import Config


class _FakePrediction:
    def __init__(self, label: str, probability: float) -> None:
        self.label = label
        self.probability = probability


class _FakeClassifier:
    """Stands in for the torch classifier: records how it was built."""

    instances: list["_FakeClassifier"] = []

    def __init__(self, checkpoint_path=None, backbone="densenet121", image_size=224, labels=None):
        self.checkpoint_path = checkpoint_path
        self.backbone = backbone
        self.labels = list(labels) if labels is not None else []
        self.model = object()
        _FakeClassifier.instances.append(self)

    def predict(self, source):
        # First label is "present" (0.9), the rest below threshold (0.1),
        # returned high-to-low like the real classifier.
        preds = [_FakePrediction(self.labels[0], 0.9)]
        preds += [_FakePrediction(l, 0.1) for l in self.labels[1:]]
        return preds


class _FakeExplanation:
    def __init__(self):
        self.centroid = (0.1, 0.1)  # upper-right / right-frontal corner
        self.overlay_png = b"\x89PNG-fake"


class _FakeExplainer:
    def __init__(self, model, backbone="densenet121", method="gradcam", **kwargs):
        self.model = model

    def explain(self, source, label, class_idx):
        return _FakeExplanation()


@pytest.fixture(autouse=True)
def _stub_engine(monkeypatch):
    _FakeClassifier.instances = []
    monkeypatch.setattr(pipeline_mod, "Classifier", _FakeClassifier)
    monkeypatch.setattr(pipeline_mod, "Explainer", _FakeExplainer)
    # No DICOM sniffing for plain-bytes sources in these tests.
    monkeypatch.setattr(pipeline_mod, "dicom_modality_tag", lambda source: None)


def test_default_modality_is_chest_xray():
    pipe = MirrorPipeline(Config())
    result = pipe.analyze(b"fake-image")
    assert result.modality == "Chest X-ray"
    assert result.meta["modality_key"] == "chest_xray"
    assert result.meta["num_labels"] == 14
    assert result.findings[0].label == "Atelectasis"  # first chest label, "present"


def test_brain_mri_routes_to_neuro_labels_and_plane():
    pipe = MirrorPipeline(Config())
    result = pipe.analyze(b"fake-image", modality="brain MRI")
    assert result.modality == "Brain MRI"
    assert result.meta["modality_key"] == "brain_mri"
    assert result.meta["num_labels"] == 11
    top = result.findings[0]
    assert top.label == "Glioma" and top.present
    # Axial plane vocabulary, not chest "zones".
    assert top.location == "the right frontal region"
    # Report is grounded in neuro language.
    assert "intracranial" in result.report.lower() or "Glioma" in result.report


def test_head_ct_routes_to_ct_taxonomy():
    pipe = MirrorPipeline(Config())
    result = pipe.analyze(b"fake-image", modality="CT")
    assert result.meta["modality_key"] == "head_ct"
    assert result.findings[0].label == "Intracranial_Hemorrhage"
    assert result.findings[0].location == "the right frontal region"


def test_engine_is_built_once_per_modality_and_cached():
    pipe = MirrorPipeline(Config())  # eagerly builds the default engine
    n_after_init = len(_FakeClassifier.instances)
    pipe.analyze(b"img", modality="brain MRI")  # builds brain engine (+1)
    pipe.analyze(b"img", modality="brain MRI")  # cache hit (+0)
    pipe.analyze(b"img", modality="chest X-ray")  # cache hit on default (+0)
    assert len(_FakeClassifier.instances) == n_after_init + 1


def test_predictions_invariant_to_posthoc_layers():
    pipe = MirrorPipeline(Config())
    src = b"fake-image"
    full = {f.label: f.probability for f in pipe.analyze(src, modality="CT").findings}
    base = {
        f.label: f.probability
        for f in pipe.analyze(src, modality="CT", localize=False, report=False).findings
    }
    assert full == base  # layers 2/3 never perturb the predictions


def test_localize_and_report_toggles_recover_ablation_conditions():
    pipe = MirrorPipeline(Config())
    src = b"fake-image"
    base = pipe.analyze(src, localize=False, report=False)
    assert all(f.overlay_png_b64 is None for f in base.findings)
    assert base.report == "" and base.report_backend == "disabled"

    full = pipe.analyze(src, localize=True, report=True)
    assert any(f.overlay_png_b64 for f in full.findings)
    assert full.report and full.report_backend in {"template", "anthropic"}


def test_per_modality_checkpoint_is_selected():
    config = Config()
    config.model.checkpoints = {"brain_mri": "weights/brain.pt"}
    config.model.checkpoint_path = "weights/chest.pt"
    pipe = MirrorPipeline(config)
    pipe.analyze(b"img", modality="brain MRI")
    by_labels = {tuple(c.labels)[0]: c for c in _FakeClassifier.instances}
    assert by_labels["Glioma"].checkpoint_path == "weights/brain.pt"
    assert by_labels["Atelectasis"].checkpoint_path == "weights/chest.pt"
