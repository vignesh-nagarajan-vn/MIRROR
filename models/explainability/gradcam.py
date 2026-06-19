"""Grad-CAM for MIRROR.

Implements Grad-CAM (Selvaraju et al., 2017) with forward/backward hooks on a
target layer. Works for both convolutional backbones (DenseNet, EfficientNet)
and — with a reshape transform — ViT, where the activations are a sequence of
patch tokens rather than a spatial grid.

The output is a normalised heatmap in [0, 1] at the spatial resolution of the
target layer, ready to be upsampled and overlaid on the input image.
"""

from __future__ import annotations

import numpy as np

try:
    import torch
    import torch.nn.functional as F
except ImportError:  # pragma: no cover
    torch = None
    F = None


class GradCAM:
    """Gradient-weighted Class Activation Mapping."""

    def __init__(self, model, target_layer, reshape_transform=None) -> None:
        if torch is None:
            raise RuntimeError("PyTorch is required for Grad-CAM.")
        self.model = model
        self.target_layer = target_layer
        self.reshape_transform = reshape_transform
        self._activations = None
        self._gradients = None
        self._fwd_handle = target_layer.register_forward_hook(self._save_activation)
        self._bwd_handle = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, _module, _inp, output):
        self._activations = output.detach()

    def _save_gradient(self, _module, _grad_in, grad_out):
        self._gradients = grad_out[0].detach()

    def _format(self, tensor):
        """Reshape ViT token sequences into a spatial grid if needed."""
        if self.reshape_transform is not None:
            return self.reshape_transform(tensor)
        return tensor

    def __call__(self, input_tensor, class_idx: int | None = None) -> np.ndarray:
        """Return a (H, W) heatmap normalised to [0, 1].

        ``class_idx`` selects which output channel to explain. If None, the
        highest-scoring class is used.
        """
        self.model.zero_grad()
        input_tensor = input_tensor.requires_grad_(True)
        logits = self.model(input_tensor)

        if class_idx is None:
            class_idx = int(logits[0].argmax().item())
        score = logits[0, class_idx]
        score.backward(retain_graph=True)

        activations = self._format(self._activations)
        gradients = self._format(self._gradients)

        # Global-average-pool the gradients to get per-channel importance weights.
        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        cam = F.interpolate(
            cam,
            size=input_tensor.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        cam = cam[0, 0].cpu().numpy()
        cam -= cam.min()
        denom = cam.max() if cam.max() > 0 else 1.0
        return cam / denom

    def remove_hooks(self) -> None:
        self._fwd_handle.remove()
        self._bwd_handle.remove()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.remove_hooks()


def vit_reshape_transform(tensor, height: int = 14, width: int = 14):
    """Reshape ViT token activations (B, N, C) into (B, C, H, W).

    Drops the leading CLS token, then folds the remaining patch tokens back into
    a 14x14 grid (for ViT-B/16 at 224px).
    """
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
    return result.permute(0, 3, 1, 2)
