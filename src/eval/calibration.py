"""
src/eval/calibration.py

25 Aug 2026 -- Expected Calibration Error (ECE) and reliability diagrams.

ECE measures whether a model's confidence matches its actual accuracy:
bucket predictions by confidence, compare per-bucket accuracy to
per-bucket average confidence, weight by bucket size. A well-calibrated
model's confidence should match how often it's actually right.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """
    Computes Expected Calibration Error.

    Parameters
    ----------
    probs : np.ndarray
        Shape (n_samples, n_classes), predicted class probabilities
        (softmax output).
    labels : np.ndarray
        Shape (n_samples,), true class indices.
    n_bins : int, default=10
        Number of confidence buckets.

    Returns
    -------
    float
        ECE in [0, 1]. 0 = perfectly calibrated, higher = more
        overconfident or underconfident.
    """
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels).astype(np.float64)

    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(labels)

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == 0:
            in_bin = (confidences >= lo) & (confidences <= hi)
        else:
            in_bin = (confidences > lo) & (confidences <= hi)
        bin_size = np.sum(in_bin)
        if bin_size == 0:
            continue
        bin_acc = np.mean(accuracies[in_bin])
        bin_conf = np.mean(confidences[in_bin])
        ece += (bin_size / n) * abs(bin_acc - bin_conf)

    return float(ece)


def plot_reliability_diagram(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
    save_path: str = None,
    title: str = "Reliability Diagram",
) -> None:
    """
    Plots a reliability diagram (per-bucket accuracy vs confidence).

    Parameters
    ----------
    probs : np.ndarray
        Shape (n_samples, n_classes), predicted class probabilities.
    labels : np.ndarray
        Shape (n_samples,), true class indices.
    n_bins : int, default=10
        Number of confidence buckets.
    save_path : str, optional
        If given, saves the figure to this path.
    title : str, default="Reliability Diagram"
        Plot title.

    Returns
    -------
    None
    """
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels).astype(np.float64)

    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_accs = []

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == 0:
            in_bin = (confidences >= lo) & (confidences <= hi)
        else:
            in_bin = (confidences > lo) & (confidences <= hi)
        if np.sum(in_bin) == 0:
            bin_accs.append(0)
        else:
            bin_accs.append(np.mean(accuracies[in_bin]))

    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
    plt.bar(bin_centers, bin_accs, width=1.0 / n_bins, alpha=0.7,
            edgecolor="black", label="Model accuracy")
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.legend()
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Saved reliability diagram to: {save_path}")
    plt.close()