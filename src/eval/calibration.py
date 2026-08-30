"""
src/eval/calibration.py

Expected Calibration Error (ECE) and reliability diagrams.
ECE measures whether a model's confidence matches its actual accuracy.
"""

from __future__ import annotations
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt

def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """Computes Expected Calibration Error."""
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels).astype(np.float64)

    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(labels)

    for i in range(n_bins):
        in_bin = (confidences > bin_edges[i]) & (confidences <= bin_edges[i+1])
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            acc_in_bin = np.mean(accuracies[in_bin])
            conf_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(conf_in_bin - acc_in_bin) * prop_in_bin

    return float(ece)

def plot_reliability_diagram(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10, save_path: str = "reliability_diagram.png"):
    """Plots and saves the reliability diagram."""
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels).astype(np.float64)

    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_accs = []
    
    for i in range(n_bins):
        in_bin = (confidences > bin_edges[i]) & (confidences <= bin_edges[i+1])
        if np.any(in_bin):
            bin_accs.append(np.mean(accuracies[in_bin]))
        else:
            bin_accs.append(0.0)

    ece = compute_ece(probs, labels, n_bins=n_bins)

    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], "--", color="gray", label="Perfect Calibration")
    plt.bar(bin_edges[:-1], bin_accs, width=1.0/n_bins, align='edge', alpha=0.6, edgecolor="black", label="Outputs")
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.title(f"Reliability Diagram (ECE = {ece:.2f})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    print(f"Smoke Test Completed.")
    print(f"Saved reliability diagram to: {save_path}")
    print(f"ECE = {ece:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Calibration Evaluation")
    parser.add_argument("--smoke-test", action="store_true", help="Run calibration smoke test")
    args = parser.parse_args()

    if args.smoke_test:
        np.random.seed(42)
        n_samples = 1000
        n_classes = 5

        # Synthetic mock data showing overconfidence (ECE ~ 0.17)
        probs = np.full((n_samples, n_classes), 0.1 / (n_classes - 1))
        probs[:, 0] = 0.90  # 90% confident on class 0

        # True labels match ~73% of the time
        labels = np.where(np.random.rand(n_samples) < 0.73, 0, 1)

        plot_reliability_diagram(probs, labels, n_bins=10, save_path="reliability_diagram.png")
    else:
        print("Please run with --smoke-test")