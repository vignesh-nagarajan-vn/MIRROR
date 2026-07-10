"""The MIRROR end-to-end pipeline.

This is the orchestration layer that realises the project's core idea:

    Image
      -> Prediction            (classification layer)
      -> Evidence Localization (explainability layer)
      -> Clinical Reasoning    (report generation layer)
      -> Human-Readable Report

Calling ``MirrorPipeline.analyze(image)`` runs all four stages and returns a
single structured result containing the predictions, per-finding overlays, and
the drafted report. Everything the backend serves and the frontend renders comes
out of this one object.
"""

from __future__ import annotations

import base64
import time
from dataclasses import dataclass, field

from .common.config import Config
from .common.modalities import ModalitySpec, resolve_modality, modality_from_dicom_tag
from .common.dicom import dicom_modality_tag
from .classification.infer import Classifier
from .explainability.explainer import Explainer, describe_location
from .report_generation.generator import ReportGenerator


@dataclass
class _ModalityEngine:
    """The classifier + explainer built for one modality (lazily, then cached)."""

    classifier: Classifier
    explainer: Explainer


@dataclass
class FindingResult:
    label: str
    probability: float
    present: bool
    location: str
    overlay_png_b64: str | None = None  # base64 PNG of the saliency overlay


@dataclass
class AnalysisResult:
    findings: list[FindingResult]
    report: str
    report_backend: str
    modality: str
    backbone: str
    explain_method: str
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "modality": self.modality,
            "backbone": self.backbone,
            "explain_method": self.explain_method,
            "report": self.report,
            "report_backend": self.report_backend,
            "findings": [
                {
                    "label": f.label,
                    "probability": round(f.probability, 4),
                    "present": f.present,
                    "location": f.location,
                    "overlay_png_b64": f.overlay_png_b64,
                }
                for f in self.findings
            ],
            "meta": self.meta,
        }


class MirrorPipeline:
    """Load the layers once per modality, then analyze images on demand.

    The classifier + explainer are modality-specific (the head width and label
    set differ between a chest X-ray, a brain MRI, and a head CT), so they are
    built lazily and cached per modality in ``self._engines``. The default
    (chest X-ray) engine is built eagerly so the backend can report ``backbone``
    and stay backward compatible. The report generator is modality-agnostic and
    shared.
    """

    def __init__(self, config: Config | None = None) -> None:
        self.config = config or Config()
        self.reporter = ReportGenerator(self.config.report)
        self._engines: dict[str, _ModalityEngine] = {}

        # Eagerly build the default modality so ``pipeline.classifier`` /
        # ``pipeline.labels`` / ``pipeline.explainer`` remain available.
        default_spec = resolve_modality(None)
        default_engine = self._engine_for(default_spec)
        self.classifier = default_engine.classifier
        self.explainer = default_engine.explainer
        self.labels = self.classifier.labels

    def _checkpoint_for(self, spec: ModalitySpec) -> str | None:
        """Resolve the checkpoint for a modality (per-modality map, then default)."""
        checkpoints = self.config.model.checkpoints or {}
        if spec.key in checkpoints:
            return checkpoints[spec.key]
        if spec.key == resolve_modality(None).key:
            return self.config.model.checkpoint_path
        return None

    def _engine_for(self, spec: ModalitySpec) -> _ModalityEngine:
        """Get (or lazily build and cache) the classifier + explainer for ``spec``."""
        engine = self._engines.get(spec.key)
        if engine is not None:
            return engine

        classifier = Classifier(
            checkpoint_path=self._checkpoint_for(spec),
            backbone=self.config.model.backbone,
            image_size=self.config.data.image_size,
            labels=spec.labels,
        )
        explainer = Explainer(
            model=classifier.model,
            backbone=self.config.model.backbone,
            method=self.config.explain.method,
            target_layer=self.config.explain.target_layer,
            image_size=self.config.data.image_size,
            overlay_alpha=self.config.explain.overlay_alpha,
            colormap=self.config.explain.colormap,
        )
        engine = _ModalityEngine(classifier=classifier, explainer=explainer)
        self._engines[spec.key] = engine
        return engine

    def _resolve_modality(self, source, modality: str | None) -> ModalitySpec:
        """Resolve the modality string, sniffing a DICOM Modality tag when 'auto'.

        Passing ``modality="auto"`` (or leaving it empty) routes by the DICOM
        (0008,0060) Modality tag when the source is DICOM, otherwise falls back to
        the default modality. An explicit modality string always wins.
        """
        if modality and str(modality).strip().lower() not in ("", "auto"):
            return resolve_modality(modality)
        tag = dicom_modality_tag(source)
        key = modality_from_dicom_tag(tag)
        return resolve_modality(key)

    def analyze(
        self,
        source,
        modality: str = "chest X-ray",
        indication: str | None = None,
        explain_top_k: int = 3,
        localize: bool = True,
        report: bool = True,
    ) -> AnalysisResult:
        """Run the Image -> Prediction -> Evidence -> Report pipeline.

        ``modality`` selects the label set / classifier head and the anatomical
        vocabulary (see ``models/common/modalities.py``); pass ``"auto"`` to route
        a DICOM study by its Modality tag. The two later layers can be switched off
        to recover MIRROR's ablation conditions, which is what
        ``evaluation/ablation.py`` compares:

        * ``localize=False, report=False`` -> classification-only baseline.
        * ``localize=True,  report=False`` -> classification + evidence.
        * ``localize=True,  report=True``  -> full MIRROR (the default).

        Skipping a layer never changes the predictions: layers 2 and 3 are
        post-hoc, so predictive metrics are identical across conditions. Per-stage
        wall-clock timings are recorded in ``meta['timings_ms']`` so the ablation
        can report the latency cost of each added layer.
        """
        spec = self._resolve_modality(source, modality)
        engine = self._engine_for(spec)
        classifier, explainer = engine.classifier, engine.explainer
        labels = classifier.labels
        threshold = self.config.report.confidence_threshold
        timings_ms: dict[str, float] = {}

        # 1. Prediction
        t0 = time.perf_counter()
        predictions = classifier.predict(source)
        timings_ms["prediction"] = (time.perf_counter() - t0) * 1000

        # 2 & 3. Evidence localisation + structured findings.
        t0 = time.perf_counter()
        findings: list[FindingResult] = []
        explained = 0
        for pred in predictions:
            present = pred.probability >= threshold
            location = "n/a"
            overlay_b64 = None

            # Only spend compute explaining the top-k present findings, and only
            # when the localisation layer is enabled.
            if localize and present and explained < explain_top_k:
                class_idx = labels.index(pred.label)
                explanation = explainer.explain(source, pred.label, class_idx)
                location = describe_location(explanation.centroid, spec.plane)
                overlay_b64 = base64.b64encode(explanation.overlay_png).decode()
                explained += 1

            findings.append(
                FindingResult(
                    label=pred.label,
                    probability=pred.probability,
                    present=present,
                    location=location,
                    overlay_png_b64=overlay_b64,
                )
            )
        timings_ms["localization"] = (time.perf_counter() - t0) * 1000

        # 4. Clinical reasoning -> report.
        t0 = time.perf_counter()
        if report:
            evidence = [
                {
                    "label": f.label,
                    "probability": f.probability,
                    "present": f.present,
                    "location": f.location,
                }
                for f in findings
            ]
            generated = self.reporter.generate(evidence, spec, indication)
            report_text, report_backend = generated.text, generated.backend
        else:
            report_text, report_backend = "", "disabled"
        timings_ms["report"] = (time.perf_counter() - t0) * 1000

        return AnalysisResult(
            findings=findings,
            report=report_text,
            report_backend=report_backend,
            modality=spec.display_name,
            backbone=self.config.model.backbone,
            explain_method=self.config.explain.method,
            meta={
                "modality_key": spec.key,
                "num_present": sum(1 for f in findings if f.present),
                "num_labels": len(labels),
                "timings_ms": timings_ms,
                "stages": {
                    "prediction": True,
                    "localization": bool(localize),
                    "report": bool(report),
                },
            },
        )
