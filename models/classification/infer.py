"""Inference wrapper around a trained MIRROR classifier.

Loads a checkpoint once and exposes ``predict()`` returning per-label
probabilities. If no checkpoint is supplied it builds an ImageNet-pretrained
backbone so the rest of the pipeline (and the demo) still runs end-to-end with
untrained — but structurally valid — outputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

from ..common.constants import CHESTXRAY14_LABELS
from ..common.preprocessing import preprocess
from .model import build_model, load_checkpoint


@dataclass
class Prediction:
    label: str
    probability: float


class Classifier:
    """Stateful classifier: load once, call ``predict`` many times.

    ``labels`` sets the finding taxonomy (and therefore the head width). It
    defaults to the 14 ChestX-ray14 labels so existing callers are unchanged; the
    pipeline passes a per-modality label set (see ``models/common/modalities.py``)
    to build a brain-MRI or head-CT classifier instead.
    """

    def __init__(
        self,
        checkpoint_path: str | None = None,
        backbone: str = "densenet121",
        device: str | None = None,
        image_size: int = 224,
        labels: Sequence[str] | None = None,
    ) -> None:
        if torch is None:
            raise RuntimeError("PyTorch is required for inference.")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.backbone = backbone
        self.image_size = image_size
        self.labels = list(labels) if labels is not None else list(CHESTXRAY14_LABELS)

        self.model = build_model(
            backbone=backbone,
            num_classes=len(self.labels),
            pretrained=(checkpoint_path is None),
        )
        if checkpoint_path and Path(checkpoint_path).exists():
            load_checkpoint(self.model, checkpoint_path, device=self.device)
        self.model.to(self.device).eval()

    @torch.no_grad() if torch else (lambda f: f)
    def predict(self, source) -> list[Prediction]:
        """Return a list of (label, probability), sorted high to low."""
        tensor, _ = preprocess(source, self.image_size)
        logits = self.model(tensor.to(self.device))
        probs = torch.sigmoid(logits)[0].cpu().tolist()
        preds = [Prediction(l, float(p)) for l, p in zip(self.labels, probs)]
        return sorted(preds, key=lambda p: p.probability, reverse=True)

    def predict_dict(self, source) -> dict[str, float]:
        return {p.label: p.probability for p in self.predict(source)}
