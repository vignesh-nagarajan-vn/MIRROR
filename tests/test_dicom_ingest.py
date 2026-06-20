"""Tests for DICOM ingestion (``models/common/dicom.py``).

These build a small DICOM in memory so they need only pydicom + numpy + Pillow
(no torch), and assert the parts of the presentation pipeline that are easy to
get silently wrong: magic-byte detection, the modality/VOI rescale, MONOCHROME1
inversion, and non-PHI metadata extraction.

Run:  pytest tests/test_dicom_ingest.py
"""

from __future__ import annotations

from io import BytesIO

import numpy as np
import pytest

pydicom = pytest.importorskip("pydicom")
from pydicom.dataset import Dataset, FileMetaDataset  # noqa: E402
from pydicom.uid import (  # noqa: E402
    ExplicitVRLittleEndian,
    SecondaryCaptureImageStorage,
    generate_uid,
)

from models.common.dicom import is_dicom, read_dicom  # noqa: E402
from models.common.preprocessing import load_image  # noqa: E402


def _make_dicom_bytes(photometric: str = "MONOCHROME1", size: int = 32) -> bytes:
    """A minimal valid DICOM: bright square in the centre, dark border."""
    img = np.full((size, size), 1000, dtype=np.uint16)
    img[size // 4 : 3 * size // 4, size // 4 : 3 * size // 4] = 4000
    if photometric == "MONOCHROME1":
        # Invert so the *stored* centre is dark; read_dicom must flip it back.
        img = img.max() - img

    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian

    ds = Dataset()
    ds.file_meta = meta
    ds.is_little_endian = True
    ds.is_implicit_VR = False
    ds.Modality = "CR"
    ds.BodyPartExamined = "CHEST"
    ds.ViewPosition = "PA"
    ds.PatientName = "TEST^PATIENT"      # PHI — must NOT appear in metadata
    ds.PatientID = "PHI-123"
    ds.Rows, ds.Columns = size, size
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = photometric
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.PixelSpacing = [0.14, 0.14]
    ds.PixelData = img.astype(np.uint16).tobytes()

    buf = BytesIO()
    ds.save_as(buf, write_like_original=False)
    return buf.getvalue()


def test_is_dicom_detects_magic_bytes():
    data = _make_dicom_bytes()
    assert is_dicom(data) is True
    assert is_dicom(b"\x89PNG\r\n\x1a\n" + b"\x00" * 200) is False


def test_is_dicom_by_extension(tmp_path):
    p = tmp_path / "scan.dcm"
    p.write_bytes(_make_dicom_bytes())
    assert is_dicom(p) is True
    assert is_dicom(str(p)) is True


def test_read_dicom_returns_rgb_and_metadata():
    di = read_dicom(_make_dicom_bytes())
    assert di.image.mode == "RGB"
    assert di.image.size == (32, 32)
    assert di.metadata["modality"] == "CR"
    assert di.metadata["body_part"] == "CHEST"
    assert di.metadata["pixel_spacing_mm"] == [0.14, 0.14]


def test_metadata_excludes_phi():
    di = read_dicom(_make_dicom_bytes())
    flat = " ".join(str(v) for v in di.metadata.values())
    assert "TEST" not in flat and "PHI-123" not in flat


def test_monochrome1_is_inverted():
    """After ingest, the centre square should be brighter than the border."""
    di = read_dicom(_make_dicom_bytes(photometric="MONOCHROME1"))
    arr = np.asarray(di.image.convert("L"), dtype=float)
    centre = arr[12:20, 12:20].mean()
    border = arr[:4, :4].mean()
    assert centre > border


def test_monochrome2_not_inverted():
    di = read_dicom(_make_dicom_bytes(photometric="MONOCHROME2"))
    arr = np.asarray(di.image.convert("L"), dtype=float)
    assert arr[12:20, 12:20].mean() > arr[:4, :4].mean()


def test_load_image_routes_dicom_from_bytes():
    """The API path: raw bytes with no filename must still be decoded as DICOM."""
    img = load_image(_make_dicom_bytes())
    assert img.mode == "RGB"
    assert img.size == (32, 32)
