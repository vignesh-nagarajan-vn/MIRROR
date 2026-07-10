"""Generate small *synthetic* brain-MRI and head-CT sample sets.

MIRROR is multi-modality (chest X-ray, brain MRI, head CT), but the real neuro
datasets (Brain Tumor MRI Dataset, RSNA Intracranial Hemorrhage) are large and
license-restricted, so they cannot be committed here. To keep the demo, the DICOM
ingest path, and smoke tests runnable out of the box for the neuro modalities,
this script fabricates a handful of axial-head-*like* images plus one DICOM per
modality carrying the correct ``Modality`` tag (``MR`` / ``CT``), so the
pipeline's DICOM auto-routing has something real to route.

    <out>/<modality>/
        images/          synthetic .png slices + one .dcm (auto-routes by tag)
        labels.csv       Image Index, Finding Labels (from the modality taxonomy)
        train_val_list.txt
        test_list.txt

These are NOT real scans and carry no diagnostic meaning; they exist only to
exercise the plumbing. The finding vocabularies are pulled from
``models/common/modalities.py`` so the labels stay in sync with the registry.

Run:
    python -m datasets.scripts.make_synthetic_neuro_samples \
        --out datasets/samples -n 12
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import numpy as np

try:
    from PIL import Image
    import pydicom
    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid, SecondaryCaptureImageStorage
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Pillow and pydicom are required: pip install pillow pydicom") from exc

from models.common.modalities import BRAIN_MRI, HEAD_CT, ModalitySpec

IMG_SIZE = 256


def _base_head(rng: np.random.Generator, modality_key: str) -> np.ndarray:
    """A crude axial 'head': skull ring, brain parenchyma, dark CSF/ventricles.

    For CT the skull (bone) is bright and the parenchyma mid-gray; for MRI the
    skull is dim and the parenchyma mid-gray with dark CSF, roughly mimicking the
    contrast a reader expects from each modality.
    """
    y, x = np.mgrid[0:IMG_SIZE, 0:IMG_SIZE].astype(np.float32)
    cx, cy = IMG_SIZE / 2, IMG_SIZE / 2
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

    skull_r = IMG_SIZE * 0.42
    brain_r = IMG_SIZE * 0.37
    img = np.full((IMG_SIZE, IMG_SIZE), 6.0, dtype=np.float32)  # air outside

    brain = r <= brain_r
    img[brain] = 90.0  # parenchyma
    # Skull rim between brain_r and skull_r.
    rim = (r > brain_r) & (r <= skull_r)
    img[rim] = 230.0 if modality_key == "head_ct" else 40.0

    # Ventricles: a small dark butterfly near the centre.
    vent = np.exp(-(((x - cx) / (IMG_SIZE * 0.06)) ** 2 + ((y - cy) / (IMG_SIZE * 0.10)) ** 2))
    img[brain] -= (55.0 * vent)[brain]

    img += rng.normal(0, 4, size=img.shape)
    return img


def _add_lesion(img: np.ndarray, rng: np.random.Generator, intensity: float, radius: float) -> None:
    """Drop a rounded lesion somewhere inside the parenchyma."""
    y, x = np.mgrid[0:IMG_SIZE, 0:IMG_SIZE].astype(np.float32)
    cx, cy = IMG_SIZE / 2, IMG_SIZE / 2
    ang = rng.uniform(0, 2 * np.pi)
    dist = rng.uniform(IMG_SIZE * 0.08, IMG_SIZE * 0.22)
    bx, by = cx + dist * np.cos(ang), cy + dist * np.sin(ang)
    img += intensity * np.exp(-(((x - bx) ** 2 + (y - by) ** 2) / (2 * radius ** 2)))


def _synth_image(rng: np.random.Generator, spec: ModalitySpec, findings: list[str]) -> np.ndarray:
    img = _base_head(rng, spec.key)
    for f in findings:
        # Tumours/haemorrhage show as bright lesions; infarcts as dark ones.
        if "Infarct" in f:
            _add_lesion(img, rng, intensity=-45, radius=rng.uniform(12, 22))
        else:
            _add_lesion(img, rng, intensity=70, radius=rng.uniform(8, 18))
    return np.clip(img, 0, 255).astype(np.uint8)


def _sample_findings(rng: random.Random, spec: ModalitySpec) -> str:
    """Pipe-delimited Finding Labels string, biased toward a normal study."""
    if rng.random() < 0.45:
        return "No Finding"
    k = rng.randint(1, 2)
    return "|".join(rng.sample(list(spec.labels), k))


def _write_dicom(path: Path, pixels: np.ndarray, spec: ModalitySpec) -> None:
    """Write an uncompressed DICOM tagged with the modality's DICOM Modality value.

    CT stores signed values with a Hounsfield rescale (intercept -1024) so the
    reader's modality LUT has work to do; MR stores plain MONOCHROME2.
    """
    dicom_modality = spec.dicom_modalities[0]  # "MR" or "CT"
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian

    ds = Dataset()
    ds.file_meta = meta
    ds.is_little_endian = True
    ds.is_implicit_VR = False
    ds.SOPClassUID = SecondaryCaptureImageStorage
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.Modality = dicom_modality
    ds.BodyPartExamined = "BRAIN" if dicom_modality == "MR" else "HEAD"
    ds.StudyDescription = f"SYNTHETIC {spec.display_name.upper()} (not a real scan)"
    ds.PatientID = "SYNTH-NEURO-0001"           # synthetic, no real PHI
    ds.PatientName = "SYNTHETIC^SAMPLE"
    ds.Rows, ds.Columns = pixels.shape
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 16
    ds.HighBit = 15
    ds.PixelSpacing = [0.5, 0.5]

    if dicom_modality == "CT":
        # Map 0..255 display to roughly -1024..+1000 HU via a rescale intercept.
        ds.BitsStored = 16
        ds.PixelRepresentation = 1  # signed
        stored = (pixels.astype(np.int16) * 8) - 1024
        ds.RescaleIntercept = -1024
        ds.RescaleSlope = 1
        ds.WindowCenter = 40
        ds.WindowWidth = 120
        # Store raw stored values (before intercept) as signed 16-bit.
        ds.PixelData = (pixels.astype(np.int16) * 8).astype(np.int16).tobytes()
    else:
        ds.BitsStored = 16
        ds.PixelRepresentation = 0  # unsigned
        stored = pixels.astype(np.uint16) * 257
        ds.WindowCenter = int(stored.mean())
        ds.WindowWidth = int(max(1, stored.max() - stored.min()))
        ds.PixelData = stored.tobytes()

    ds.save_as(str(path), write_like_original=False)


def _generate_one(spec: ModalitySpec, out_root: Path, num: int, seed: int, test_frac: float) -> None:
    out = out_root / spec.key
    images_dir = out / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    np_rng = np.random.default_rng(seed)
    py_rng = random.Random(seed)

    rows = []
    n_test = max(1, int(num * test_frac))
    prefix = "mri" if spec.key == "brain_mri" else "ct"
    for i in range(num):
        findings = _sample_findings(py_rng, spec)
        active = [] if findings == "No Finding" else findings.split("|")
        pixels = _synth_image(np_rng, spec, active)

        is_dicom = (i == 0)
        name = f"{prefix}_{i:04d}.dcm" if is_dicom else f"{prefix}_{i:04d}.png"
        if is_dicom:
            _write_dicom(images_dir / name, pixels, spec)
        else:
            Image.fromarray(pixels).save(images_dir / name)

        rows.append({"Image Index": name, "Finding Labels": findings})

    with open(out / "labels.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["Image Index", "Finding Labels"])
        writer.writeheader()
        writer.writerows(rows)

    names = [r["Image Index"] for r in rows]
    (out / "test_list.txt").write_text("\n".join(names[:n_test]) + "\n", encoding="utf-8")
    (out / "train_val_list.txt").write_text("\n".join(names[n_test:]) + "\n", encoding="utf-8")

    print(f"[{spec.display_name}] wrote {len(names)} samples to {images_dir} (1 DICOM: {prefix}_0000.dcm)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic brain-MRI + head-CT samples.")
    parser.add_argument("--out", default="datasets/samples")
    parser.add_argument("-n", "--num", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-frac", type=float, default=0.25)
    args = parser.parse_args()

    out_root = Path(args.out)
    for spec in (BRAIN_MRI, HEAD_CT):
        _generate_one(spec, out_root, args.num, args.seed, args.test_frac)


if __name__ == "__main__":
    main()
