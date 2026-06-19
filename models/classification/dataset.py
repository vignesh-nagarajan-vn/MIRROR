"""ChestX-ray14 dataset wrapper.

Expects the NIH release layout:

    data_root/
        images/                 # all .png files, flat
        Data_Entry_2017.csv     # official metadata
        train_val_list.txt
        test_list.txt

The CSV's ``Finding Labels`` column is a pipe-delimited string such as
``"Effusion|Infiltration"`` or ``"No Finding"``. We convert it to a 14-dim
multi-hot target vector.
"""

from __future__ import annotations

from pathlib import Path

try:
    import torch
    from torch.utils.data import Dataset
    import pandas as pd
except ImportError:  # pragma: no cover
    torch = None
    Dataset = object  # type: ignore
    pd = None

from ..common.constants import CHESTXRAY14_LABELS
from ..common.preprocessing import build_transform, load_image


class ChestXray14Dataset(Dataset):
    """Multi-label dataset for NIH ChestX-ray14."""

    def __init__(
        self,
        data_root: str | Path,
        split: str = "train",
        image_size: int = 224,
    ) -> None:
        if pd is None:
            raise RuntimeError("pandas and torch are required for the dataset.")
        self.data_root = Path(data_root)
        self.image_dir = self.data_root / "images"
        self.split = split
        self.transform = build_transform(image_size, train=(split == "train"))
        self.label_index = {name: i for i, name in enumerate(CHESTXRAY14_LABELS)}

        meta = pd.read_csv(self.data_root / "Data_Entry_2017.csv")
        split_file = "test_list.txt" if split == "test" else "train_val_list.txt"
        with open(self.data_root / split_file, "r", encoding="utf-8") as fh:
            allowed = {line.strip() for line in fh if line.strip()}
        self.records = meta[meta["Image Index"].isin(allowed)].reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.records)

    def _encode_labels(self, finding_labels: str):
        target = torch.zeros(len(CHESTXRAY14_LABELS), dtype=torch.float32)
        if finding_labels and finding_labels != "No Finding":
            for label in finding_labels.split("|"):
                if label in self.label_index:
                    target[self.label_index[label]] = 1.0
        return target

    def __getitem__(self, idx: int):
        row = self.records.iloc[idx]
        image = load_image(self.image_dir / row["Image Index"])
        tensor = self.transform(image)
        target = self._encode_labels(row["Finding Labels"])
        return tensor, target
