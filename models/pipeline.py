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
from .classification.infer import Classifier
from .explainability.explainer import Explainer, describe_location
from .report_generation.generator import ReportGenerator


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
    """Load all three layers once, then analyze images on demand."""

    def __init__(self, config: Config | None = None) -> None:
        self.config = config or Config()

        self.classifier = Classifier(
            checkpoint_path=self.config.model.checkpoint_path,
            backbone=self.config.model.backbone,
            image_size=self.config.data.image_size,
        )
        self.explainer = Explainer(
            model=self.classifier.model,
            backbone=self.config.model.backbone,
            method=self.config.explain.method,
            target_layer=self.config.explain.target_layer,
            image_size=self.config.data.image_size,
            overlay_alpha=self.config.explain.overlay_alpha,
            colormap=self.config.explain.colormap,
        )
        self.reporter = ReportGenerator(self.config.report)
        self.labels = self.classifier.labels

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

        The two later layers can be switched off to recover MIRROR's ablation
        conditions, which is what ``evaluation/ablation.py`` compares:

        * ``localize=False, report=False`` -> classification-only baseline.
        * ``localize=True,  report=False`` -> classification + evidence.
        * ``localize=True,  report=True``  -> full MIRROR (the default).

        Skipping a layer never changes the predictions: layers 2 and 3 are
        post-hoc, so predictive metrics are identical across conditions. Per-stage
        wall-clock timings are recorded in ``meta['timings_ms']`` so the ablation
        can report the latency cost of each added layer.
        """
        threshold = self.config.report.confidence_threshold
        timings_ms: dict[str, float] = {}

        # 1. Prediction
        t0 = time.perf_counter()
        predictions = self.classifier.predict(source)
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
                class_idx = self.labels.index(pred.label)
                explanation = self.explainer.explain(source, pred.label, class_idx)
                location = describe_location(explanation.centroid)
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
            generated = self.reporter.generate(evidence, modality, indication)
            report_text, report_backend = generated.text, generated.backend
        else:
            report_text, report_backend = "", "disabled"
        timings_ms["report"] = (time.perf_counter() - t0) * 1000

        return AnalysisResult(
            findings=findings,
            report=report_text,
            report_backend=report_backend,
            modality=modality,
            backbone=self.config.model.backbone,
            explain_method=self.config.explain.method,
            meta={
                "num_present": sum(1 for f in findings if f.present),
                "timings_ms": timings_ms,
                "stages": {
                    "prediction": True,
                    "localization": bool(localize),
                    "report": bool(report),
                },
            },
        )
