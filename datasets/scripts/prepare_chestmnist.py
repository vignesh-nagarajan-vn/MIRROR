"""Prepare ChestMNIST (MedMNIST v2) in the NIH ChestX-ray14 directory layout.

ChestMNIST is a real, openly-licensed (CC BY 4.0) benchmark derived from the NIH
ChestX-ray14 release: the **same 14 pathologies, in the same label order** as
``CHESTXRAY14_LABELS``, multi-label, with official train/val/test splits, just
downsampled. That makes it a legitimate small-scale stand-in for the full 45 GB
release: MIRROR's taxonomy applies unchanged, and the numbers are real (real
radiographs, real labels) — only the resolution and, here, the training budget are
reduced.

This script downloads ChestMNIST via the ``medmnist`` package and writes a subset
to disk in exactly the layout ``ChestXray14Dataset`` expects, so the existing
train/evaluate/ablation pipeline runs on it with **no code changes**:

    <out>/
        images/                 # <split>_<idx>.png grayscale frames
        Data_Entry_2017.csv     # Image Index, Finding Labels (pipe-delimited)
        train_val_list.txt
        test_list.txt

Run:
    python -m datasets.scripts.prepare_chestmnist --out datasets/raw/chestmnist \
        --n-train-val 8000 --n-test 12000 --size 64 --seed 42
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

try:
    from PIL import Image
    from medmnist import ChestMNIST
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Pillow and medmnist are required: pip install pillow medmnist") from exc

from models.common.constants import CHESTXRAY14_LABELS


def _finding_labels(multi_hot: np.ndarray) -> str:
    """Turn a 14-dim 0/1 vector into the pipe-delimited NIH 'Finding Labels'."""
    active = [CHESTXRAY14_LABELS[i] for i in range(len(CHESTXRAY14_LABELS)) if multi_hot[i]]
    return "|".join(active) if active else "No Finding"


def _load_split(split: str, size: int):
    """Load a ChestMNIST split, trying the requested size then falling back to 28."""
    try:
        ds = ChestMNIST(split=split, download=True, size=size)
    except Exception as exc:  # noqa: BLE001 - fall back to the base 28px release
        print(f"  [warn] size={size} unavailable ({exc}); falling back to 28px")
        ds = ChestMNIST(split=split, download=True)
    return ds.imgs, ds.labels  # imgs: (N,H,W) uint8 ; labels: (N,14) 0/1


def _export(imgs, labels, names_out, rows_out, images_dir, prefix, count, rng):
    n = min(count, len(imgs)) if count else len(imgs)
    idx = rng.permutation(len(imgs))[:n]
    for j in idx:
        name = f"{prefix}_{int(j):06d}.png"
        arr = imgs[int(j)]
        if arr.ndim == 3:  # (H,W,1) -> (H,W)
            arr = arr[..., 0]
        Image.fromarray(arr.astype(np.uint8), mode="L").save(images_dir / name)
        names_out.append(name)
        rows_out.append({"Image Index": name,
                         "Finding Labels": _finding_labels(labels[int(j)])})
    return n


def main() -> None:
    p = argparse.ArgumentParser(description="Prepare ChestMNIST in NIH layout.")
    p.add_argument("--out", default="datasets/raw/chestmnist")
    p.add_argument("--n-train-val", type=int, default=8000,
                   help="Images to export for training+validation (0 = all ~89k).")
    p.add_argument("--n-test", type=int, default=12000,
                   help="Images to export for the test split (0 = all 22,433).")
    p.add_argument("--size", type=int, default=64, choices=[28, 64, 128, 224])
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    out = Path(args.out)
    images_dir = out / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    rows: list[dict] = []
    train_val_names: list[str] = []
    test_names: list[str] = []

    # Pool the official train + val into "train_val" (the pipeline holds out its own
    # val); keep the official test split separate and untouched for evaluation.
    print(f"Loading ChestMNIST (size={args.size}) ...")
    tr_imgs, tr_lbls = _load_split("train", args.size)
    va_imgs, va_lbls = _load_split("val", args.size)
    te_imgs, te_lbls = _load_split("test", args.size)

    tv_imgs = np.concatenate([tr_imgs, va_imgs], axis=0)
    tv_lbls = np.concatenate([tr_lbls, va_lbls], axis=0)

    n_tv = _export(tv_imgs, tv_lbls, train_val_names, rows, images_dir, "trv",
                   args.n_train_val, rng)
    n_te = _export(te_imgs, te_lbls, test_names, rows, images_dir, "tst",
                   args.n_test, rng)

    with open(out / "Data_Entry_2017.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["Image Index", "Finding Labels"])
        writer.writeheader()
        writer.writerows(rows)
    (out / "train_val_list.txt").write_text("\n".join(train_val_names) + "\n", encoding="utf-8")
    (out / "test_list.txt").write_text("\n".join(test_names) + "\n", encoding="utf-8")

    print(f"Wrote {n_tv} train_val + {n_te} test images to {images_dir}")
    print(f"  metadata: {out/'Data_Entry_2017.csv'}")


if __name__ == "__main__":
    main()
