"""DICOM ingestion for MIRROR.

Real radiology data is delivered as DICOM (``.dcm``), not PNG/JPEG. A raw DICOM
pixel array is *not* display-ready: it may be stored as 12/16-bit signed
integers, may need a rescale slope/intercept (modality LUT), a window
center/width (VOI LUT), and may be photometrically inverted (MONOCHROME1, where
high values are black). Feeding the raw array straight into an ImageNet-normalised
classifier would silently destroy contrast.

This module turns a DICOM source into the same kind of 8-bit RGB ``PIL.Image``
the rest of the pipeline already understands, applying the standard presentation
pipeline along the way, and extracts a small set of *non-PHI* technical tags that
can enrich the report (modality, view position, body part, laterality).

Decoding compressed transfer syntaxes (JPEG/JPEG2000) requires an extra handler
(``pylibjpeg`` or ``gdcm``); uncompressed DICOM needs only numpy. The functions
here degrade gracefully and raise a clear error if a handler is missing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path

import numpy as np

try:
    import pydicom
    from pydicom.pixel_data_handlers.util import apply_modality_lut, apply_voi_lut
    from PIL import Image
except ImportError:  # pragma: no cover - optional until DICOM is used
    pydicom = None
    apply_modality_lut = None
    apply_voi_lut = None
    Image = None

# DICOM files carry the magic token "DICM" at byte offset 128 (after the
# 128-byte preamble). Extension is a weaker hint since many archives drop it.
_DICOM_MAGIC_OFFSET = 128
_DICOM_MAGIC = b"DICM"
DICOM_EXTENSIONS: tuple[str, ...] = (".dcm", ".dicom", ".dco")

# Technical tags worth surfacing. Deliberately excludes patient identifiers so we
# never lift PHI out of the file and into a report or log.
_SAFE_METADATA_TAGS: tuple[tuple[str, str], ...] = (
    ("Modality", "modality"),
    ("BodyPartExamined", "body_part"),
    ("ViewPosition", "view_position"),
    ("ImageLaterality", "laterality"),
    ("Laterality", "laterality"),
    ("StudyDescription", "study_description"),
    ("SeriesDescription", "series_description"),
    ("PhotometricInterpretation", "photometric_interpretation"),
)


@dataclass
class DicomImage:
    """A display-ready DICOM image plus its non-PHI technical metadata."""

    image: "Image.Image"            # 8-bit RGB, ready for the standard transform
    metadata: dict = field(default_factory=dict)


def is_dicom(source: str | Path | bytes) -> bool:
    """Best-effort DICOM detection by magic bytes, then by extension.

    Works on both a path and an in-memory byte buffer (the API receives bytes).
    """
    if isinstance(source, (str, Path)):
        path = Path(source)
        if path.suffix.lower() in DICOM_EXTENSIONS:
            return True
        try:
            with open(path, "rb") as fh:
                head = fh.read(_DICOM_MAGIC_OFFSET + len(_DICOM_MAGIC))
        except OSError:
            return False
        return _has_dicom_magic(head)
    return _has_dicom_magic(source[: _DICOM_MAGIC_OFFSET + len(_DICOM_MAGIC)])


def _has_dicom_magic(head: bytes) -> bool:
    return (
        len(head) >= _DICOM_MAGIC_OFFSET + len(_DICOM_MAGIC)
        and head[_DICOM_MAGIC_OFFSET : _DICOM_MAGIC_OFFSET + len(_DICOM_MAGIC)]
        == _DICOM_MAGIC
    )


def _extract_metadata(ds) -> dict:
    meta: dict = {}
    for tag, key in _SAFE_METADATA_TAGS:
        value = getattr(ds, tag, None)
        if value not in (None, "") and key not in meta:
            meta[key] = str(value)
    spacing = getattr(ds, "PixelSpacing", None)
    if spacing is not None:
        # Stored as [row_mm, col_mm]; keep as floats so downstream code can turn
        # pixel measurements into millimetres.
        try:
            meta["pixel_spacing_mm"] = [float(spacing[0]), float(spacing[1])]
        except (TypeError, ValueError, IndexError):
            pass
    if getattr(ds, "Rows", None) and getattr(ds, "Columns", None):
        meta["pixel_dims"] = [int(ds.Rows), int(ds.Columns)]
    return meta


def _to_display_array(ds) -> np.ndarray:
    """Apply the standard DICOM presentation pipeline -> 8-bit grayscale array."""
    arr = ds.pixel_array
    # Multi-frame (e.g. a CT/MRI series in one file): take the middle slice so a
    # 2D classifier still gets a representative, well-centred image.
    if arr.ndim == 3 and getattr(ds, "SamplesPerPixel", 1) == 1:
        arr = arr[arr.shape[0] // 2]

    arr = arr.astype(np.float32)
    if apply_modality_lut is not None:
        # Rescale slope/intercept (e.g. CT -> Hounsfield units).
        arr = apply_modality_lut(arr, ds).astype(np.float32)
    if apply_voi_lut is not None:
        try:
            # Window center/width or an explicit VOI LUT, if present.
            arr = apply_voi_lut(arr, ds).astype(np.float32)
        except Exception:  # noqa: BLE001 - malformed VOI LUT: fall back to raw range
            pass

    # MONOCHROME1: high stored value = black. Invert so bright = high, matching
    # MONOCHROME2 and ordinary images.
    if str(getattr(ds, "PhotometricInterpretation", "")).upper() == "MONOCHROME1":
        arr = arr.max() - arr

    lo, hi = float(arr.min()), float(arr.max())
    if hi > lo:
        arr = (arr - lo) / (hi - lo)
    else:
        arr = np.zeros_like(arr)
    return (arr * 255.0).astype(np.uint8)


def read_dicom(source: str | Path | bytes) -> DicomImage:
    """Decode a DICOM source into a display-ready RGB image plus metadata.

    Accepts a filesystem path or raw bytes. Raises a clear ``RuntimeError`` if
    pydicom is unavailable or the pixel data needs a handler that isn't installed.
    """
    if pydicom is None or Image is None:
        raise RuntimeError(
            "pydicom and Pillow are required to read DICOM. "
            "Install with: pip install pydicom pillow"
        )

    reader = BytesIO(source) if isinstance(source, bytes) else str(source)
    # force=True tolerates files written without the 128-byte preamble/DICM magic.
    ds = pydicom.dcmread(reader, force=True)

    try:
        gray = _to_display_array(ds)
    except (AttributeError, NotImplementedError, ValueError) as exc:
        raise RuntimeError(
            "Could not decode DICOM pixel data. Compressed transfer syntaxes "
            "(JPEG/JPEG2000) require an extra handler — install 'pylibjpeg "
            "pylibjpeg-libjpeg' or 'python-gdcm'."
        ) from exc

    image = Image.fromarray(gray).convert("RGB")
    return DicomImage(image=image, metadata=_extract_metadata(ds))
