"""Tests that report generation is modality-aware (``models/report_generation``).

Torch-free: the template backend and the prompt builder consume only the
structured evidence, so no ML stack is needed. These pin the fix for the original
bug where a brain study was described with chest phrasing ("no acute
cardiopulmonary abnormality").

Run:  pytest tests/test_report_modality.py
"""

from __future__ import annotations

from models.common.modalities import resolve_modality
from models.report_generation.generator import ReportGenerator, _template_report, DISCLAIMER
from models.report_generation.prompts import build_user_prompt


def _absent(spec) -> list[dict]:
    return [
        {"label": l, "probability": 0.05, "present": False, "location": "n/a"}
        for l in spec.labels
    ]


def test_normal_chest_uses_cardiopulmonary_phrasing():
    spec = resolve_modality("chest X-ray")
    report = _template_report(_absent(spec), spec)
    assert "No acute cardiopulmonary abnormality detected." in report


def test_normal_brain_does_not_use_chest_phrasing():
    spec = resolve_modality("brain MRI")
    report = _template_report(_absent(spec), spec)
    assert "No acute intracranial abnormality detected." in report
    assert "cardiopulmonary" not in report.lower()


def test_normal_head_ct_uses_intracranial_phrasing():
    spec = resolve_modality("CT")
    report = _template_report(_absent(spec), spec)
    assert "No acute intracranial abnormality detected." in report


def test_disclaimer_is_ascii_only():
    # The em dash used to garble on Windows cp1252 consoles; the disclaimer must
    # stay ASCII so the CLI demo prints cleanly everywhere.
    assert DISCLAIMER.isascii()
    spec = resolve_modality("brain MRI")
    assert _template_report(_absent(spec), spec).isascii()


def test_generate_accepts_both_spec_and_string():
    spec = resolve_modality("brain MRI")
    findings = _absent(spec)
    gen = ReportGenerator()  # provider defaults to anthropic -> template fallback (no key)
    by_spec = gen.generate(findings, spec).text
    by_string = gen.generate(findings, "brain MRI").text
    assert by_spec == by_string
    assert "intracranial" in by_spec.lower()


def test_user_prompt_carries_modality_and_guidance():
    spec = resolve_modality("brain MRI")
    prompt = build_user_prompt(_absent(spec), spec)
    assert "Modality: Brain MRI" in prompt
    assert "neuroanatomy" in prompt  # from the modality's report_guidance
    assert "Glioma" in prompt        # evidence block lists the taxonomy
