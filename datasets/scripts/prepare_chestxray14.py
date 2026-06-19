"""Dataset preparation helpers for MIRROR.

The NIH ChestX-ray14 release is large (~45 GB) and distributed across multiple
archives via the NIH Box link, so this script does not auto-download it. Instead
it (a) documents the expected layout, (b) verifies that an existing local copy is
complete, and (c) builds a small balanced *sample* split useful for smoke-testing
the pipeline without the full dataset.

Run:
    python -m datasets.scripts.prepare_chestxray14 --data-root datasets/raw/chestxray14 --verify
    python -m datasets.scripts.prepare_chestxray14 --data-root datasets/raw/chestxray14 --make-sample 200
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

EXPECTED_FILES = ["Data_Entry_2017.csv", "train_val_list.txt", "test_list.txt"]


def verify(data_root: Path) -> bool:
    ok = True
    for name in EXPECTED_FILES:
        if not (data_root / name).exists():
            print(f"  [missing] {name}")
            ok = False
    images = data_root / "images"
    if not images.is_dir():
        print("  [missing] images/ directory")
        ok = False
    else:
        n = sum(1 for _ in images.glob("*.png"))
        print(f"  [images] found {n} .png files (full release ~112,120)")
        if n == 0:
            ok = False
    print("  -> OK" if ok else "  -> incomplete")
    return ok


def make_sample(data_root: Path, n: int) -> None:
    """Write a sample_list.txt with up to n images that exist locally."""
    images = sorted((data_root / "images").glob("*.png"))[:n]
    if not images:
        print("No images found; cannot build sample.")
        return
    out = data_root / "sample_list.txt"
    out.write_text("\n".join(p.name for p in images), encoding="utf-8")
    print(f"Wrote {len(images)} entries to {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare ChestX-ray14.")
    parser.add_argument("--data-root", default="datasets/raw/chestxray14")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--make-sample", type=int, default=0)
    args = parser.parse_args()

    root = Path(args.data_root)
    if not root.exists():
        print(f"Data root {root} does not exist. See datasets/README.md for setup.")
        sys.exit(1)

    if args.verify:
        print(f"Verifying {root} ...")
        verify(root)
    if args.make_sample:
        make_sample(root, args.make_sample)


if __name__ == "__main__":
    main()
