"""Clinical report generator.

Two backends:

* ``anthropic`` — calls the Claude API to draft a free-text report from the
  structured evidence. Requires ``ANTHROPIC_API_KEY`` in the environment.
* ``template`` — a fully offline, deterministic generator that fills a report
  skeleton from the same evidence. This guarantees MIRROR produces a coherent
  report even with no network/API access, which is important for the demo and
  for reproducible evaluation.

Both consume the identical structured evidence, so swapping backends never
changes what the report is allowed to say — only how fluently it says it.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..common.config import ReportConfig, get_anthropic_api_key
from ..common.modalities import ModalitySpec, resolve_modality
from .prompts import SYSTEM_PROMPT, build_user_prompt

# ASCII hyphen (not an em dash) so the deterministic report prints cleanly on
# any terminal, including Windows cp1252 consoles.
DISCLAIMER = "AI-GENERATED DRAFT - requires verification by a licensed radiologist."


@dataclass
class Report:
    text: str
    backend: str
    findings_used: list[dict]


def _qualitative_confidence(p: float) -> str:
    if p >= 0.85:
        return "high confidence"
    if p >= 0.65:
        return "moderate confidence"
    return "low confidence"


def _template_report(findings: list[dict], spec: ModalitySpec) -> str:
    """Offline deterministic report from structured evidence.

    Modality-aware: the "normal" impression phrasing comes from the modality spec
    so a normal brain study is not described as "no acute cardiopulmonary
    abnormality".
    """
    present = [f for f in findings if f["present"]]
    absent_major = [
        f["label"] for f in findings if not f["present"]
    ][:4]

    lines = [f"FINDINGS:"]
    if present:
        for f in present:
            conf = _qualitative_confidence(f["probability"])
            lines.append(
                f"- {f['label'].replace('_', ' ')} identified in "
                f"{f.get('location', 'an unspecified region')} ({conf}, "
                f"p={f['probability']:.2f})."
            )
    else:
        lines.append("- No acute findings above the reporting threshold.")
    if absent_major:
        negs = ", ".join(l.replace("_", " ") for l in absent_major)
        lines.append(f"- No evidence of: {negs}.")

    lines.append("")
    lines.append("IMPRESSION:")
    if present:
        top = max(present, key=lambda f: f["probability"])
        lines.append(
            f"1. {top['label'].replace('_', ' ')} is the dominant finding "
            f"({_qualitative_confidence(top['probability'])})."
        )
        for i, f in enumerate(
            sorted(present, key=lambda f: f["probability"], reverse=True)[1:], start=2
        ):
            lines.append(f"{i}. Possible {f['label'].replace('_', ' ')}.")
    else:
        lines.append(f"1. {spec.normal_impression}")

    lines.append("")
    lines.append(DISCLAIMER)
    return "\n".join(lines)


class ReportGenerator:
    """Generate a clinician-style report from structured evidence."""

    def __init__(self, config: ReportConfig | None = None) -> None:
        self.config = config or ReportConfig()

    def _anthropic_report(self, findings: list[dict], spec: ModalitySpec,
                          indication: str | None) -> str:
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "The 'anthropic' package is required for the anthropic backend."
            ) from exc

        api_key = get_anthropic_api_key()
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set.")

        client = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(
            model=self.config.model,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            system=SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": build_user_prompt(findings, spec, indication),
                }
            ],
        )
        return "".join(
            block.text for block in message.content if block.type == "text"
        )

    def generate(
        self,
        findings: list[dict],
        modality: ModalitySpec | str = "chest X-ray",
        indication: str | None = None,
    ) -> Report:
        """Produce a report, falling back to the template backend on any error.

        ``modality`` may be a resolved :class:`ModalitySpec` (as the pipeline
        passes) or a free-text string (resolved via the registry), so this stays
        usable standalone.
        """
        spec = (
            modality if isinstance(modality, ModalitySpec) else resolve_modality(modality)
        )
        backend = self.config.provider
        if backend == "anthropic":
            try:
                text = self._anthropic_report(findings, spec, indication)
                return Report(text=text, backend="anthropic", findings_used=findings)
            except Exception as exc:  # noqa: BLE001 - graceful degradation
                print(f"[report] anthropic backend failed ({exc}); using template.")
                backend = "template"

        text = _template_report(findings, spec)
        return Report(text=text, backend="template", findings_used=findings)
