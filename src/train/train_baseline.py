"""
PhysioShift - Baseline supervised training loop
21 Aug 2026 - Member 3 - Phase 3, Day 6

Smoke test: generates synthetic multi-domain signals, runs them through the
real harmonization pipeline (PhysioShiftDataset.process_signal), then trains
LightweightResNet1D on the resulting windows for a few epochs.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.metrics import f1_score
import wandb

from src.models.resnet1d import LightweightResNet1D
from src.harmonize.dataset import PhysioShiftDataset

# ---------------- CONFIG ----------------
CONFIG = {
    "epochs": 3,
    "batch_size": 16,
    "lr": 1e-3,
    "patience": 3,
    "num_classes": 3,       # one class per synthetic domain, for the smoke test
    "target_fs": 100.0,
    "window_sec": 10.0,
    "overlap": 0.5,
}


class WindowedSignalDataset(Dataset):
    """Wraps windowed numpy arrays + labels as a standard PyTorch Dataset."""

    def __init__(self, windows, labels):
        # windows: (num_windows, window_size) -> add channel dim -> (num_windows, 1, window_size)
        self.x = torch.tensor(windows, dtype=torch.float32).unsqueeze(1)
        self.y = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def build_smoke_test_data(config):
    """
    Generates synthetic multi-domain ECG-like signals, runs each through the
    real harmonization pipeline, and returns stacked windows + fake labels
    (one label per synthetic domain), for smoke-testing the training loop.
    """
    harmonizer = PhysioShiftDataset(
        target_fs=config["target_fs"],
        window_sec=config["window_sec"],
        overlap=config["overlap"],
    )

    rng = np.random.default_rng(0)
    fs = 250
    duration_sec = 30
    t = np.linspace(0, duration_sec, duration_sec * fs, endpoint=False)

    domain_ids = ["D1_SMOKE_domainA", "D2_SMOKE_domainB", "D3_SMOKE_domainC"]

    all_windows = []
    all_labels = []

    for label, domain_id in enumerate(domain_ids):
        freq = 1.0 + 0.3 * label  # slightly different signal per domain
        fake_signal = np.sin(2 * np.pi * freq * t) + 0.1 * rng.standard_normal(t.shape)

        windowed = harmonizer.process_signal(
            fake_signal, original_fs=fs, modality="ecg", domain_id=domain_id
        )
        all_windows.append(windowed)
        all_labels.extend([label] * len(windowed))

    x = np.concatenate(all_windows, axis=0)
    y = np.array(all_labels)
    return x, y


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            total_loss += loss.item() * x.size(0)
            preds = out.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    avg_loss = total_loss / len(loader.dataset)
    f1_macro = f1_score(all_labels, all_preds, average="macro")
    return avg_loss, f1_macro


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    wandb.init(project="physioshift-baseline", config=CONFIG)

    import numpy as np

    x, y = build_smoke_test_data(CONFIG)
    print(f"Smoke test data shape: {x.shape}, labels shape: {y.shape}")  

    full_dataset = WindowedSignalDataset(x, y)
    val_size = max(1, int(0.2 * len(full_dataset)))
    train_size = len(full_dataset) - val_size
    train_subset, val_subset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_subset, batch_size=CONFIG["batch_size"], shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=CONFIG["batch_size"], shuffle=False)

    model = LightweightResNet1D(num_classes=CONFIG["num_classes"]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"])

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, CONFIG["epochs"] + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_f1 = evaluate(model, val_loader, criterion, device)

        print(f"Epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_f1={val_f1:.4f}")
        wandb.log({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_f1_macro": val_f1})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "smoke_test_checkpoint.pt")
        else:
            patience_counter += 1
            if patience_counter >= CONFIG["patience"]:
                print("Early stopping triggered.")
                break

    wandb.finish()
    print("Smoke test complete. Check your W&B dashboard for the run link.")


if __name__ == "__main__":
    main()
