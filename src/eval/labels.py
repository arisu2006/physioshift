"""
src/eval/labels.py

25 Aug 2026 -- shared physiological label generator.

Domain identity CANNOT be used as a classification label for Split B/C,
because train and test domains are disjoint by construction -- a model
would need to predict a class it never saw during training. Instead,
this module derives a label that means the SAME thing regardless of
which dataset/device a window came from: heart rate bucket, computed
via peak detection on the (already resampled, filtered, normalized)
window.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.signal import find_peaks


LOW_BPM_THRESHOLD = 60.0
HIGH_BPM_THRESHOLD = 100.0
MIN_PLAUSIBLE_BPM = 30.0
MAX_PLAUSIBLE_BPM = 220.0

LABEL_NAMES = ["low", "normal", "high"]


def compute_hr_label(window: np.ndarray, fs: float, window_sec: float = 10.0) -> Optional[int]:
    """
    Computes a heart-rate bucket label (0=low, 1=normal, 2=high) for a
    single signal window via peak detection.

    Parameters
    ----------
    window : np.ndarray
        1D signal window (already resampled/filtered/normalized).
    fs : float
        Sampling rate of the window, in Hz.
    window_sec : float, default=10.0
        Duration of the window, in seconds.

    Returns
    -------
    int or None
        0 (low, <60bpm), 1 (normal, 60-100bpm), 2 (high, >100bpm).
        None if peak detection produced an implausible bpm (<30 or >220),
        indicating the window should be dropped rather than mislabeled.
    """
    window = np.asarray(window, dtype=np.float64).flatten()

    min_distance = max(1, int(round(fs * 60.0 / MAX_PLAUSIBLE_BPM)))

    peaks, _ = find_peaks(window, distance=min_distance, prominence=0.3)
    n_peaks = len(peaks)

    bpm = (n_peaks / window_sec) * 60.0

    if bpm < MIN_PLAUSIBLE_BPM or bpm > MAX_PLAUSIBLE_BPM:
        return None

    if bpm < LOW_BPM_THRESHOLD:
        return 0
    elif bpm <= HIGH_BPM_THRESHOLD:
        return 1
    else:
        return 2


def label_windows(windows: np.ndarray, fs: float, window_sec: float = 10.0):
    """
    Labels a batch of windows, dropping any with implausible bpm.

    Parameters
    ----------
    windows : np.ndarray
        Shape (n_windows, window_size).
    fs : float
        Sampling rate, in Hz.
    window_sec : float, default=10.0
        Window duration, in seconds.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (valid_windows, labels) -- windows with implausible peak counts
        are excluded from both arrays.
    """
    valid_windows = []
    labels = []
    for w in windows:
        label = compute_hr_label(w, fs, window_sec)
        if label is not None:
            valid_windows.append(w)
            labels.append(label)

    if not valid_windows:
        return np.empty((0, windows.shape[1])), np.empty((0,), dtype=int)

    return np.stack(valid_windows, axis=0), np.array(labels, dtype=int)