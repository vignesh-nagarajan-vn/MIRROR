"""Generate a small *synthetic* ChestX-ray14-style sample dataset.

The real NIH ChestX-ray14 release is ~45 GB and license-restricted, so it cannot
be committed to or downloaded from this repository. To make MIRROR runnable
end-to-end out of the box — demo, dataset loader, training smoke test, and the
DICOM ingest path — this script fabricates a handful of chest-X-ray-*like*
images and the exact metadata layout the real dataset uses:

    <out>/
        images/                 # synthetic .png frames + one .dcm (DICOM ingest)
        Data_Entry_2017.csv     # Image Index, Finding Labels, ...
        train_val_list.txt
        test_list.txt

These images are NOT real radiographs and carry no diagnostic meaning. They exist
only to exercise the plumbing. For real results, fetch the NIH release with
``download_chestxray14.py`` and point ``data.data_root`` at it.

Run:
    python -m datasets.scripts.make_synthetic_samples --out datasets/samples/chestxray14 -n 24
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
    raise SystemExit(
        "Pillow and pydicom are required: pip install pillow pydicom"
    ) from exc

from models.common.constants import CHESTXRAY14_LABELS

IMG_SIZE = 256


def _base_thorax(rng: np.random.Generator) -> np.ndarray:
    """A crude grayscale 'thorax': dark field, two lighter lungs, a mediastinum."""
    y, x = np.mgrid[0:IMG_SIZE, 0:IMG_SIZE].astype(np.float32)
    cx, cy = IMG_SIZE / 2, IMG_SIZE / 2
    img = np.full((IMG_SIZE, IMG_SIZE), 18.0, dtype=np.float32)

    for sign in (-1, 1):  # left and right lung fields
        lx = cx + sign * IMG_SIZE * 0.22
        ly = cy + IMG_SIZE * 0.02
        lung = np.exp(-(((x - lx) / (IMG_SIZE * 0.16)) ** 2
                        + ((y - ly) / (IMG_SIZE * 0.26)) ** 2))
        img += 120.0 * lung
    # Mediastinum / spine: a brighter central column.
    img += 80.0 * np.exp(-(((x - cx) / (IMG_SIZE * 0.05)) ** 2))
    img += rng.normal(0, 6, size=img.shape)  # sensor noise
    return img


def _add_blob(img: np.ndarray, rng, intensity: float, radius: float) -> None:
    y, x = np.mgrid[0:IMG_SIZE, 0:IMG_SIZE].astype(np.float32)
    bx = rng.uniform(IMG_SIZE * 0.2, IMG_SIZE * 0.8)
    by = rng.uniform(IMG_SIZE * 0.2, IMG_SIZE * 0.8)
    img += intensity * np.exp(-(((x - bx) ** 2 + (y - by) ** 2) / (2 * radius ** 2)))


def _synth_image(rng: np.random.Generator, findings: list[str]) -> np.ndarray:
    img = _base_thorax(rng)
    # Map a few labels to visible artefacts so overlays land on *something*.
    for f in findings:
        if f in ("Mass", "Nodule"):
            _add_blob(img, rng, intensity=90, radius=rng.uniform(6, 14))
        elif f in ("Effusion", "Infiltration", "Consolidation", "Edema"):
            _add_blob(img, rng, intensity=55, radius=rng.uniform(20, 40))
        elif f == "Cardiomegaly":
            _add_blob(img, rng, intensity=70, radius=28)
    return np.clip(img, 0, 255).astype(np.uint8)


def _sample_findings(rng: random.Random) -> str:
    """Pipe-delimited Finding Labels string, biased toward 'No Finding'."""
    if rng.random() < 0.45:
        return "No Finding"
    k = rng.randint(1, 2)
    return "|".join(rng.sample(CHESTXRAY14_LABELS, k))


def _write_dicom(path: Path, pixels: np.ndarray, findings: str) -> None:
    """Write an uncompressed MONOCHROME1 DICOM to exercise the ingest path.

    MONOCHROME1 + a window/center makes this a non-trivial decode: MIRROR's
    reader must invert it and apply the VOI LUT to recover a sensible image.
    """
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
    ds.Modality = "CR"
    ds.BodyPartExamined = "CHEST"
    ds.ViewPosition = "PA"
    ds.StudyDescription = "SYNTHETIC CHEST PA (not a real radiograph)"
    ds.PatientID = "SYNTH-0001"            # synthetic, no real PHI
    ds.PatientName = "SYNTHETIC^SAMPLE"
    ds.Rows, ds.Columns = pixels.shape
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME1"  # inverted: high = black
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.PixelSpacing = [0.143, 0.143]
    # Store inverted 16-bit so the reader's MONOCHROME1 handling has work to do.
    stored = (65535 - pixels.astype(np.uint16) * 257)
    ds.WindowCenter = int(stored.mean())
    ds.WindowWidth = int(max(1, stored.max() - stored.min()))
    ds.PixelData = stored.tobytes()
    ds.save_as(str(path), write_like_original=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic CXR14 samples.")
    parser.add_argument("--out", default="datasets/samples/chestxray14")
    parser.add_argument("-n", "--num", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-frac", type=float, default=0.25)
    args = parser.parse_args()

    out = Path(args.out)
    images_dir = out / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    np_rng = np.random.default_rng(args.seed)
    py_rng = random.Random(args.seed)

    rows = []
    n_test = max(1, int(args.num * args.test_frac))
    for i in range(args.num):
        findings = _sample_findings(py_rng)
        active = [] if findings == "No Finding" else findings.split("|")
        pixels = _synth_image(np_rng, active)

        # Make exactly one sample a DICOM so the .dcm ingest path is covered.
        is_dicom = (i == 0)
        name = f"synth_{i:04d}.dcm" if is_dicom else f"synth_{i:04d}.png"
        if is_dicom:
            _write_dicom(images_dir / name, pixels, findings)
        else:
            Image.fromarray(pixels).save(images_dir / name)

        rows.append({
            "Image Index": name,
            "Finding Labels": findings,
            "Patient ID": f"{i:05d}",
            "Patient Age": py_rng.randint(20, 85),
            "Patient Gender": py_rng.choice(["M", "F"]),
            "View Position": py_rng.choice(["PA", "AP"]),
        })

    # Metadata CSV (subset of the real Data_Entry_2017.csv columns).
    with open(out / "Data_Entry_2017.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    names = [r["Image Index"] for r in rows]
    test_names = names[:n_test]
    train_names = names[n_test:]
    (out / "test_list.txt").write_text("\n".join(test_names) + "\n", encoding="utf-8")
    (out / "train_val_list.txt").write_text("\n".join(train_names) + "\n", encoding="utf-8")

    print(f"Wrote {len(names)} synthetic samples to {images_dir}")
    print(f"  train_val: {len(train_names)}  |  test: {len(test_names)}  |  dicom: 1")
    print(f"  metadata:  {out/'Data_Entry_2017.csv'}")


if __name__ == "__main__":
    main()
