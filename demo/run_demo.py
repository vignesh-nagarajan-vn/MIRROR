"""Command-line demo for MIRROR.

Runs the complete pipeline on a single image and prints the predictions and the
draft report, saving any saliency overlays next to the input. Works with no
trained checkpoint (ImageNet weights) and with the offline template report
backend, so it runs anywhere with the Python deps installed.

Accepts PNG/JPEG/BMP/WEBP or native DICOM (.dcm) — DICOM is decoded with the
modality/VOI LUT and MONOCHROME1 inversion applied automatically.

Usage:
    python -m demo.run_demo path/to/xray.png
    python -m demo.run_demo path/to/study.dcm
    python -m demo.run_demo path/to/xray.png --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import base64
from pathlib import Path

from models.common.config import Config
from models.pipeline import MirrorPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MIRROR on one image.")
    parser.add_argument("image", help="Path to a radiograph (PNG/JPEG/BMP/WEBP/DICOM).")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--modality", default="chest X-ray")
    parser.add_argument("--indication", default=None)
    parser.add_argument("--outdir", default="demo/assets")
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        raise SystemExit(f"No such file: {image_path}")

    try:
        config = Config.from_yaml(args.config)
    except Exception:
        config = Config()

    print("Loading MIRROR pipeline (first run downloads pretrained weights)...")
    pipeline = MirrorPipeline(config)

    result = pipeline.analyze(
        image_path.read_bytes(),
        modality=args.modality,
        indication=args.indication,
    )

    print("\n=== PREDICTIONS ===")
    for f in result.findings:
        flag = "*" if f.present else " "
        print(f" {flag} {f.label:<20} {f.probability * 100:5.1f}%   {f.location}")

    print("\n=== DRAFT REPORT ===")
    print(result.report)
    print(f"\n[report backend: {result.report_backend}]")

    # Save overlays.
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    saved = 0
    for f in result.findings:
        if f.overlay_png_b64:
            out = outdir / f"{image_path.stem}_{f.label}_overlay.png"
            out.write_bytes(base64.b64decode(f.overlay_png_b64))
            saved += 1
    if saved:
        print(f"\nSaved {saved} saliency overlay(s) to {outdir}/")


if __name__ == "__main__":
    main()
