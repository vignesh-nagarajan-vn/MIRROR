"""Classification backbones for MIRROR.

A small factory that swaps between the three backbones named in the project
overview — DenseNet121, EfficientNet-B0, and ViT-B/16 — behind one interface.
Each returns a multi-label head (sigmoid is applied at loss/inference time, not
inside the module) because ChestX-ray14 is a multi-label problem: a single film
can show several findings at once.
"""

from __future__ import annotations

try:
    import torch
    import torch.nn as nn
    from torchvision import models
except ImportError:  # pragma: no cover
    torch = None
    nn = object  # type: ignore
    models = None

from ..common.constants import NUM_CHESTXRAY14_CLASSES


SUPPORTED_BACKBONES = ("densenet121", "efficientnet_b0", "vit_b_16")

# The convolutional/attention layer Grad-CAM should hook into for each backbone.
DEFAULT_TARGET_LAYERS = {
    "densenet121": "features.norm5",
    "efficientnet_b0": "features.8",
    "vit_b_16": "encoder.layers.encoder_layer_11.ln_1",
}


def build_model(
    backbone: str = "densenet121",
    num_classes: int = NUM_CHESTXRAY14_CLASSES,
    pretrained: bool = True,
    dropout: float = 0.2,
):
    """Construct a classifier with the requested backbone.

    The classifier head outputs raw logits of shape (B, num_classes). Apply
    sigmoid for multi-label probabilities.
    """
    if models is None:
        raise RuntimeError("torchvision is required to build models.")
    backbone = backbone.lower()
    if backbone not in SUPPORTED_BACKBONES:
        raise ValueError(
            f"Unknown backbone '{backbone}'. Choose from {SUPPORTED_BACKBONES}."
        )

    if backbone == "densenet121":
        weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        net = models.densenet121(weights=weights)
        in_features = net.classifier.in_features
        net.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes),
        )

    elif backbone == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        net = models.efficientnet_b0(weights=weights)
        in_features = net.classifier[1].in_features
        net.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes),
        )

    else:  # vit_b_16
        weights = models.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
        net = models.vit_b_16(weights=weights)
        in_features = net.heads.head.in_features
        net.heads = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes),
        )

    return net


def resolve_target_layer(net, backbone: str, override: str | None = None):
    """Return the nn.Module Grad-CAM should attach to.

    Resolves a dotted path like ``features.norm5`` into the actual submodule.
    """
    path = override or DEFAULT_TARGET_LAYERS[backbone.lower()]
    module = net
    for attr in path.split("."):
        module = module[int(attr)] if attr.isdigit() else getattr(module, attr)
    return module


def load_checkpoint(net, checkpoint_path: str, device: str = "cpu"):
    """Load weights from a saved checkpoint into ``net`` in place."""
    state = torch.load(checkpoint_path, map_location=device)
    # Support both raw state_dicts and {"model": state_dict, ...} wrappers.
    state_dict = state.get("model", state) if isinstance(state, dict) else state
    net.load_state_dict(state_dict)
    return net
