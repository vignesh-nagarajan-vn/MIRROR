"""Download the real NIH ChestX-ray14 dataset.

NIH publishes ChestX-ray14 as 12 image archives (``images_001.tar.gz`` ...
``images_012.tar.gz``, ~45 GB total) plus the metadata CSV and the official
train/val and test split lists, hosted on an NIH Box share. This script fetches
them, extracts the archives into a single flat ``images/`` directory, and lays
everything out exactly as ``ChestXray14Dataset`` and the prep/verify script
expect.

It downloads a lot of data — expect tens of GB and a long run. Use ``--max-archives``
to grab only the first few archives for a partial set while developing.

Run:
    python -m datasets.scripts.download_chestxray14 --data-root datasets/raw/chestxray14
    python -m datasets.scripts.download_chestxray14 --data-root datasets/raw/chestxray14 --max-archives 2
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import tarfile
import urllib.request
from pathlib import Path

# Official NIH Box direct-download links for the 12 image archives.
# Source: https://nihcc.app.box.com/v/ChestXray-NIHCC (NIH Clinical Center).
IMAGE_ARCHIVE_URLS = [
    "https://nihcc.box.com/shared/static/vfk49d74nhbxq3p5h4255maygm08kqgi.gz",
    "https://nihcc.box.com/shared/static/i28rlmbvmfjbl8p2n3ril0pptcmcu9d1.gz",
    "https://nihcc.box.com/shared/static/f1t00wrtdk94satdfb9olcolqx20z2jp.gz",
    "https://nihcc.box.com/shared/static/0aowwzs5lhjrceb3qp67ahp0rd1l1etg.gz",
    "https://nihcc.box.com/shared/static/v5e3goj22zr6h8tzualxfsqlqaygfbsn.gz",
    "https://nihcc.box.com/shared/static/asi7ikud9jwnkrnkj99jnpfkjdes7l6l.gz",
    "https://nihcc.box.com/shared/static/jn1b4mw4n6lnh74ovmcjb8y48h8xj07n.gz",
    "https://nihcc.box.com/shared/static/tvpxmn7qyrgl0w8wfh9kqfjskv6nmm1j.gz",
    "https://nihcc.box.com/shared/static/upyy3ml7qdumlgk2rfcvlb9k6gvqq2pj.gz",
    "https://nihcc.box.com/shared/static/l6nilvfa9cg3s28tqv1qc1olm3gnz54p.gz",
    "https://nihcc.box.com/shared/static/hhq8fkdgvcari67vfhs7ppg2w6ni4jze.gz",
    "https://nihcc.box.com/shared/static/ioqwiy20ihqwyr8pf4c24eazhh281pbu.gz",
]

# Metadata + official split files (name -> URL).
METADATA_URLS = {
    "Data_Entry_2017_v2020.csv": "https://nihcc.box.com/shared/static/wkflz6ll2j7nx6c8wuufdvbazm9zhf4z.csv",
    "train_val_list.txt": "https://nihcc.box.com/shared/static/ulc1q8d8vh8j7ueb0jpvslm0z7uqlb9j.txt",
    "test_list.txt": "https://nihcc.box.com/shared/static/nbb4arm38pr0pco72jmnqig9d4qm4v98.txt",
}


def _download(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  [skip] {dest.name} already present")
        return
    print(f"  [get ] {dest.name} <- {url}")
    tmp = dest.with_suffix(dest.suffix + ".part")

    def _progress(block: int, block_size: int, total: int) -> None:
        if total > 0:
            pct = min(100, block * block_size * 100 // total)
            sys.stdout.write(f"\r        {pct:3d}%")
            sys.stdout.flush()

    urllib.request.urlretrieve(url, tmp, reporthook=_progress)
    sys.stdout.write("\r")
    tmp.rename(dest)


def _extract(archive: Path, images_dir: Path) -> None:
    print(f"  [tar ] extracting {archive.name}")
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar.getmembers():
            if not member.isfile():
                continue
            # The archives store files under an "images/" prefix; flatten them.
            member.name = Path(member.name).name
            tar.extract(member, images_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download NIH ChestX-ray14.")
    parser.add_argument("--data-root", default="datasets/raw/chestxray14")
    parser.add_argument("--max-archives", type=int, default=len(IMAGE_ARCHIVE_URLS),
                        help="Download only the first N of 12 image archives.")
    parser.add_argument("--keep-archives", action="store_true",
                        help="Keep the .tar.gz files after extraction.")
    args = parser.parse_args()

    root = Path(args.data_root)
    archives_dir = root / "archives"
    images_dir = root / "images"
    archives_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading metadata + split lists ...")
    for name, url in METADATA_URLS.items():
        _download(url, root / name)
    # The dataset wrapper expects the canonical filename; provide both.
    canonical = root / "Data_Entry_2017.csv"
    versioned = root / "Data_Entry_2017_v2020.csv"
    if versioned.exists() and not canonical.exists():
        canonical.write_bytes(versioned.read_bytes())

    n = min(args.max_archives, len(IMAGE_ARCHIVE_URLS))
    print(f"Downloading {n}/{len(IMAGE_ARCHIVE_URLS)} image archives "
          f"(full set ~45 GB) ...")
    for i, url in enumerate(IMAGE_ARCHIVE_URLS[:n], start=1):
        archive = archives_dir / f"images_{i:03d}.tar.gz"
        _download(url, archive)
        _extract(archive, images_dir)
        if not args.keep_archives:
            archive.unlink(missing_ok=True)

    count = sum(1 for _ in images_dir.glob("*.png"))
    print(f"\nDone. {count} images in {images_dir} "
          f"(full release = 112,120). Verify with:")
    print(f"  python -m datasets.scripts.prepare_chestxray14 "
          f"--data-root {root} --verify")


if __name__ == "__main__":
    main()
