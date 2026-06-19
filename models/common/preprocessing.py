"""Image loading and preprocessing utilities.

Centralising the transforms guarantees that training, inference, and
explainability all see pixels normalised the same way — a common source of
silent bugs when Grad-CAM is computed on a differently-scaled tensor than the
one the classifier was trained on.
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import numpy as np

try:
    import torch
    from torchvision import transforms
    from PIL import Image
except ImportError:  # pragma: no cover
    torch = None
    transforms = None
    Image = None

from .constants import DEFAULT_IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD


def build_transform(image_size: int = DEFAULT_IMAGE_SIZE, train: bool = False):
    """Return the torchvision transform pipeline.

    Training adds light augmentation (random resized crop + horizontal flip).
    Inference is deterministic so explanations are reproducible.
    """
    if transforms is None:
        raise RuntimeError("torchvision is required for transforms.")

    normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    if train:
        return transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.RandomResizedCrop(image_size, scale=(0.85, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            normalize,
        ])

    return transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        normalize,
    ])


def load_image(source: str | Path | bytes) -> "Image.Image":
    """Load an image from a path or raw bytes into RGB PIL form."""
    if Image is None:
        raise RuntimeError("Pillow is required to load images.")
    if isinstance(source, (str, Path)):
        return Image.open(source).convert("RGB")
    return Image.open(BytesIO(source)).convert("RGB")


def preprocess(source: str | Path | bytes, image_size: int = DEFAULT_IMAGE_SIZE):
    """Load and transform an image into a (1, 3, H, W) batched tensor."""
    if torch is None:
        raise RuntimeError("torch is required for preprocessing.")
    image = load_image(source)
    tensor = build_transform(image_size, train=False)(image)
    return tensor.unsqueeze(0), image


def denormalize(tensor) -> np.ndarray:
    """Undo ImageNet normalisation, returning an HxWx3 uint8 array.

    Used to draw Grad-CAM overlays on top of the original-looking image.
    """
    mean = np.array(IMAGENET_MEAN).reshape(3, 1, 1)
    std = np.array(IMAGENET_STD).reshape(3, 1, 1)
    arr = tensor.detach().cpu().numpy()
    if arr.ndim == 4:
        arr = arr[0]
    arr = (arr * std + mean)
    arr = np.clip(arr, 0, 1)
    arr = (arr.transpose(1, 2, 0) * 255).astype(np.uint8)
    return arr
