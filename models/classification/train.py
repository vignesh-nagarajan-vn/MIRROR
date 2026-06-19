"""Training entry point for the MIRROR classifier.

Usage:
    python -m models.classification.train --config configs/default.yaml

Multi-label classification with ``BCEWithLogitsLoss``. Validation reports macro
AUROC, the standard headline metric on ChestX-ray14.
"""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    import torch
    from torch.utils.data import DataLoader, random_split
    from sklearn.metrics import roc_auc_score
    import numpy as np
except ImportError:  # pragma: no cover
    torch = None

from ..common.config import Config
from ..common.constants import CHESTXRAY14_LABELS
from .model import build_model
from .dataset import ChestXray14Dataset


def _resolve_device(requested: str) -> str:
    if requested == "cuda" and not torch.cuda.is_available():
        print("[train] CUDA requested but unavailable — falling back to CPU.")
        return "cpu"
    return requested


def evaluate(model, loader, device) -> float:
    """Return macro AUROC over the validation set."""
    model.eval()
    all_targets, all_scores = [], []
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            scores = torch.sigmoid(model(images)).cpu().numpy()
            all_scores.append(scores)
            all_targets.append(targets.numpy())
    y_true = np.concatenate(all_targets)
    y_score = np.concatenate(all_scores)
    aucs = []
    for i in range(y_true.shape[1]):
        if len(np.unique(y_true[:, i])) > 1:  # AUROC undefined for one class
            aucs.append(roc_auc_score(y_true[:, i], y_score[:, i]))
    return float(np.mean(aucs)) if aucs else 0.0


def train(config: Config) -> None:
    if torch is None:
        raise RuntimeError("PyTorch is required to train.")

    torch.manual_seed(config.train.seed)
    device = _resolve_device(config.train.device)

    full = ChestXray14Dataset(
        config.data.data_root, split="train", image_size=config.data.image_size
    )
    val_size = max(1, int(0.1 * len(full)))
    train_size = len(full) - val_size
    train_set, val_set = random_split(full, [train_size, val_size])

    train_loader = DataLoader(
        train_set,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=(device == "cuda"),
    )
    val_loader = DataLoader(
        val_set,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
    )

    model = build_model(
        backbone=config.model.backbone,
        num_classes=config.model.num_classes,
        pretrained=config.model.pretrained,
        dropout=config.model.dropout,
    ).to(device)

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.train.lr, weight_decay=config.train.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.train.epochs
    )

    output_dir = Path(config.train.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_auc = 0.0

    for epoch in range(1, config.train.epochs + 1):
        model.train()
        running = 0.0
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), targets)
            loss.backward()
            optimizer.step()
            running += loss.item()
        scheduler.step()

        val_auc = evaluate(model, val_loader, device)
        avg_loss = running / max(1, len(train_loader))
        print(f"[epoch {epoch:02d}] loss={avg_loss:.4f}  val_macroAUC={val_auc:.4f}")

        if val_auc > best_auc:
            best_auc = val_auc
            ckpt = output_dir / f"{config.model.backbone}_best.pt"
            torch.save(
                {
                    "model": model.state_dict(),
                    "backbone": config.model.backbone,
                    "labels": CHESTXRAY14_LABELS,
                    "val_macro_auc": best_auc,
                },
                ckpt,
            )
            print(f"           saved new best -> {ckpt}")

    print(f"[train] done. best val macro AUROC = {best_auc:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the MIRROR classifier.")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    config = Config.from_yaml(args.config)
    train(config)


if __name__ == "__main__":
    main()
