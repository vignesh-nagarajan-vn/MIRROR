"""Score-CAM for MIRROR.

Score-CAM (Wang et al., 2020) is a gradient-free alternative to Grad-CAM. Each
activation map is upsampled, used to mask the input, and the resulting change in
the target class score becomes that map's weight. It tends to produce cleaner,
less noisy localisations at the cost of extra forward passes.
"""

from __future__ import annotations

import numpy as np

try:
    import torch
    import torch.nn.functional as F
except ImportError:  # pragma: no cover
    torch = None
    F = None


class ScoreCAM:
    """Gradient-free Class Activation Mapping via score perturbation."""

    def __init__(self, model, target_layer, batch_size: int = 16) -> None:
        if torch is None:
            raise RuntimeError("PyTorch is required for Score-CAM.")
        self.model = model
        self.target_layer = target_layer
        self.batch_size = batch_size
        self._activations = None
        self._handle = target_layer.register_forward_hook(self._save_activation)

    def _save_activation(self, _module, _inp, output):
        self._activations = output.detach()

    @torch.no_grad() if torch else (lambda f: f)
    def __call__(self, input_tensor, class_idx: int | None = None) -> np.ndarray:
        device = input_tensor.device
        logits = self.model(input_tensor)
        if class_idx is None:
            class_idx = int(logits[0].argmax().item())

        activations = self._activations  # (1, C, h, w)
        _, channels, _, _ = activations.shape
        h, w = input_tensor.shape[-2:]

        # Upsample every activation map to input resolution and min-max scale it.
        upsampled = F.interpolate(
            activations, size=(h, w), mode="bilinear", align_corners=False
        )[0]  # (C, H, W)
        scores = torch.zeros(channels, device=device)

        for start in range(0, channels, self.batch_size):
            end = min(start + self.batch_size, channels)
            masks = upsampled[start:end]
            mins = masks.flatten(1).min(dim=1)[0].view(-1, 1, 1)
            maxs = masks.flatten(1).max(dim=1)[0].view(-1, 1, 1)
            norm = (masks - mins) / (maxs - mins + 1e-8)
            masked = input_tensor * norm.unsqueeze(1)  # (b, 3, H, W)
            out = torch.softmax(self.model(masked), dim=1)[:, class_idx]
            scores[start:end] = out

        weights = torch.softmax(scores, dim=0).view(channels, 1, 1)
        cam = (weights * upsampled).sum(dim=0)
        cam = F.relu(cam).cpu().numpy()
        cam -= cam.min()
        denom = cam.max() if cam.max() > 0 else 1.0
        return cam / denom

    def remove_hooks(self) -> None:
        self._handle.remove()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.remove_hooks()
