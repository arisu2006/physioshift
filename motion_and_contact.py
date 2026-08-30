"""
motion_and_contact.py  (Day 27 -- Member 3)
----------------------------------------------
Two hardware-informed corruption functions:

  1. motion_artifact_burst()
     WHY it looks the way it does: when a wearable shifts on the skin
     (arm swing, wrist rotation, walking), the sensor-skin contact
     geometry changes abruptly for a short window. For PPG this shows
     up as a large-amplitude, broadband burst that swamps the true
     pulsatile signal for ~0.3-2s, then contact re-settles and the
     signal returns to baseline. It's not random noise -- it's a
     localized high-amplitude, high-frequency burst.

  2. poor_contact_dropout()
     WHY it looks the way it does: when electrode-skin (ECG) or
     optode-skin (PPG) contact is marginal (loose strap, sweat,
     movement), impedance spikes and the effective coupling to the
     sensor drops. The result isn't noise added on top of the signal --
     it's signal *attenuation* toward the sensor's floor/rail value,
     sometimes with the true signal barely visible underneath, for a
     sustained period until contact is restored.

Both take a clean 1D signal window and return a corrupted copy, so they
compose cleanly into PhysioShiftAugmentor later (30 Aug).
"""

import numpy as np
import matplotlib.pyplot as plt


def motion_artifact_burst(
    signal: np.ndarray,
    fs: int = 100,
    burst_duration_range=(0.3, 2.0),
    amplitude_factor: float = 4.0,
    seed=None,
):
    """
    Injects a localized high-amplitude, broadband burst simulating
    sudden sensor-skin displacement (e.g. arm swing).

    Args:
        signal: 1D array, the clean harmonized window
        fs: sampling rate in Hz (matches the harmonization target rate)
        burst_duration_range: (min_sec, max_sec) burst length
        amplitude_factor: how many x the signal's own std the burst spikes to
        seed: for reproducible testing

    Returns:
        corrupted: 1D array, same length as `signal`
    """
    rng = np.random.default_rng(seed)
    corrupted = signal.copy().astype(np.float32)
    n = len(signal)

    burst_len = int(rng.uniform(*burst_duration_range) * fs)
    burst_len = min(burst_len, n)
    if burst_len <= 0:
        return corrupted

    start = rng.integers(0, max(1, n - burst_len))
    end = start + burst_len

    sig_std = np.std(signal) if np.std(signal) > 1e-8 else 1.0

    # Broadband burst: high-frequency noise envelope-modulated so it
    # ramps in, peaks, and settles back out (not a hard rectangular edge --
    # real contact loss/regain isn't instantaneous)
    t = np.linspace(0, 1, burst_len)
    envelope = np.sin(np.pi * t) ** 2  # smooth ramp up/down, peak at center
    broadband_noise = rng.normal(0, 1, size=burst_len)
    # add a lower-frequency wander component too -- real motion artifacts
    # aren't pure high-frequency noise, there's a baseline-shift component
    low_freq = np.sin(2 * np.pi * rng.uniform(0.5, 3.0) * t)

    burst = amplitude_factor * sig_std * envelope * (0.7 * broadband_noise + 0.3 * low_freq)
    corrupted[start:end] += burst

    return corrupted


def poor_contact_dropout(
    signal: np.ndarray,
    fs: int = 100,
    dropout_duration_range=(1.0, 4.0),
    attenuation_range=(0.05, 0.3),
    floor_noise_std: float = 0.05,
    seed=None,
):
    """
    Simulates sustained poor electrode/optode-skin contact: the true
    signal is attenuated toward a near-flat floor value (not replaced
    with noise -- contact loss suppresses amplitude, it doesn't add
    energy), for a longer sustained period than a motion burst.

    Args:
        signal: 1D array, the clean harmonized window
        fs: sampling rate in Hz
        dropout_duration_range: (min_sec, max_sec) dropout length
        attenuation_range: fraction of original amplitude retained (e.g. 0.05-0.3)
        floor_noise_std: small residual sensor-floor noise during dropout
        seed: for reproducible testing

    Returns:
        corrupted: 1D array, same length as `signal`
    """
    rng = np.random.default_rng(seed)
    corrupted = signal.copy().astype(np.float32)
    n = len(signal)

    dropout_len = int(rng.uniform(*dropout_duration_range) * fs)
    dropout_len = min(dropout_len, n)
    if dropout_len <= 0:
        return corrupted

    start = rng.integers(0, max(1, n - dropout_len))
    end = start + dropout_len

    attenuation = rng.uniform(*attenuation_range)

    # smooth onset/recovery of contact loss rather than a hard cut,
    # since impedance rises/falls gradually, not instantly
    t = np.linspace(0, 1, dropout_len)
    onset_recovery = 1 - np.sin(np.pi * t) ** 2 * (1 - attenuation)

    corrupted[start:end] *= onset_recovery
    corrupted[start:end] += rng.normal(0, floor_noise_std, size=dropout_len)

    return corrupted


# ---------------------------------------------------------------------------
# Visual validation: before/after plots with hardware-cause captions
# ---------------------------------------------------------------------------
def make_synthetic_signal(n=1000, fs=100, seed=0):
    """Stand-in for a real harmonized PPG/ECG window (swap for real data)."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, n / fs, n)
    clean = np.sin(2 * np.pi * 1.2 * t) + 0.3 * np.sin(2 * np.pi * 2.4 * t)
    clean += rng.normal(0, 0.02, size=n)
    return clean.astype(np.float32)


def plot_before_after(clean, corrupted, title, caption, save_path):
    fig, axes = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
    axes[0].plot(clean, color="#4C72B0")
    axes[0].set_title(f"{title} -- Before")
    axes[0].set_ylabel("Amplitude")

    axes[1].plot(corrupted, color="#C44E52")
    axes[1].set_title(f"{title} -- After")
    axes[1].set_ylabel("Amplitude")
    axes[1].set_xlabel("Sample")

    fig.suptitle(caption, fontsize=9, y=0.02)
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved {save_path}")


def main():
    clean = make_synthetic_signal(seed=42)

    burst = motion_artifact_burst(clean, fs=100, seed=1)
    plot_before_after(
        clean, burst,
        title="Motion Artifact Burst",
        caption="Cause: sudden sensor-skin displacement (arm swing) -- broadband amplitude burst, ramps in/out over ~0.3-2s.",
        save_path="motion_artifact_burst.png",
    )

    dropout = poor_contact_dropout(clean, fs=100, seed=2)
    plot_before_after(
        clean, dropout,
        title="Poor Contact Dropout",
        caption="Cause: rising electrode/optode-skin impedance from loose contact -- sustained amplitude attenuation toward the sensor floor.",
        save_path="poor_contact_dropout.png",
    )

    # sanity checks -- confirm the corrupted output actually deviates
    assert not np.allclose(clean, burst), "burst had no effect"
    assert not np.allclose(clean, dropout), "dropout had no effect"
    print("Both functions verified: outputs differ from clean input as expected.")


if __name__ == "__main__":
    main()
