"""
src/train/train_split.py

25 Aug 2026 -- Phase 3, Day 6 (Member 1)

Generic training/eval script for ANY split (A, B, or C) using the
shared heart-rate-bucket label (src/eval/labels.py), so results are
comparable and meaningful across all three splits -- unlike domain_id,
this label space is identical between train and test, so Split B/C
(unseen device / clinical->consumer) are genuine generalization tests.

Usage:
    python -m src.train.train_split --split a
    python -m src.train.train_split --split b
    python -m src.train.train_split --split c
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch
import torch.nn as nn
from augmentor import PhysioShiftAugmentor
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score

from src.models.resnet1d import LightweightResNet1D
from src.eval.splits import (
    load_harmonized_cache,
    split_a_in_device,
    split_b_unseen_device,
    split_c_clinical_to_consumer,
)
from src.eval.labels import label_windows
from src.eval.calibration import compute_ece, plot_reliability_diagram

TARGET_FS = 100.0
NUM_CLASSES = 3  # low / normal / high heart rate

CONFIG = {
    "epochs": 5,
    "batch_size": 16,
    "lr": 1e-3,
}


class WindowedLabelDataset(Dataset):
    """Wraps windowed numpy arrays + HR-bucket labels with optional augmentation."""

    def __init__(self, windows: np.ndarray, labels: np.ndarray, augmentor=None, fs=100):
        self.windows = windows.astype(np.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.augmentor = augmentor
        self.fs = fs

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sig = self.windows[idx]
        if self.augmentor is not None:
            sig, _ = self.augmentor(sig, fs=self.fs)
        x = torch.tensor(sig, dtype=torch.float32).unsqueeze(0)  # (1, window_size)
        y = self.labels[idx]
        return x, y


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)


def evaluate(model, loader, device):
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            out = model(x)
            probs = torch.softmax(out, dim=1).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(y.numpy())
    probs = np.concatenate(all_probs, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    preds = np.argmax(probs, axis=1)
    f1 = f1_score(labels, preds, average="macro")
    return f1, probs, labels


def main(split_name: str):
    cache = load_harmonized_cache()

    if split_name == "a":
        split = split_a_in_device(cache)
    elif split_name == "b":
        split = split_b_unseen_device(cache)
    elif split_name == "c":
        split = split_c_clinical_to_consumer(cache)
    else:
        raise ValueError(f"Unknown split: {split_name}")

    train_windows, train_labels = label_windows(split.train_windows, fs=TARGET_FS)
    test_windows, test_labels = label_windows(split.test_windows, fs=TARGET_FS)

    print(f"Split {split_name.upper()}: {len(train_labels)} labeled train windows, "
          f"{len(test_labels)} labeled test windows")

    if len(train_labels) == 0 or len(test_labels) == 0:
        print(f"[Split {split_name.upper()}] Not enough labeled windows to train. Skipping.")
        return None

    augmentor = PhysioShiftAugmentor(p_augment=0.5)
    train_ds = WindowedLabelDataset(train_windows, train_labels, augmentor=augmentor, fs=TARGET_FS)
    test_ds = WindowedLabelDataset(test_windows, test_labels, augmentor=None, fs=TARGET_FS)
    train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"], shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=CONFIG["batch_size"], shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LightweightResNet1D(num_classes=NUM_CLASSES, in_channels=1, base_channels=48).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"])
    criterion = nn.CrossEntropyLoss()

    for epoch in range(CONFIG["epochs"]):
        loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        print(f"[Split {split_name.upper()}] Epoch {epoch+1}/{CONFIG['epochs']} -- loss: {loss:.4f}")

    f1, probs, labels = evaluate(model, test_loader, device)
    ece = compute_ece(probs, labels)

    print(f"\n[Split {split_name.upper()}] Test F1-macro: {f1:.4f} | ECE: {ece:.4f}")

    os.makedirs("results/checkpoints", exist_ok=True)
    torch.save(model.state_dict(), f"results/checkpoints/split_{split_name}_baseline.pt")

    os.makedirs("results/reliability_diagrams", exist_ok=True)
    plot_reliability_diagram(
        probs, labels,
        save_path=f"results/reliability_diagrams/split_{split_name}_reliability.png",
    )

    return f1, ece


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, required=True, choices=["a", "b", "c"])
    args = parser.parse_args()
    main(args.split)