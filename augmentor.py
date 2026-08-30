"""
augmentor.py — PhysioShiftAugmentor
Combines corruption functions into a unified pipeline.
"""

import numpy as np
import random

# Import functions from Day 27
try:
    from motion_and_contact import motion_artifact_burst, poor_contact_dropout
except ModuleNotFoundError:
    try:
        from src.augment.motion_and_contact import motion_artifact_burst, poor_contact_dropout
    except ModuleNotFoundError:
        def motion_artifact_burst(signal, **kwargs):
            corrupted = signal.copy()
            idx = np.random.randint(0, max(1, len(signal) - 50))
            corrupted[idx:idx+50] += np.random.normal(0, 2.0, size=min(50, len(signal)-idx))
            return corrupted

        def poor_contact_dropout(signal, **kwargs):
            corrupted = signal.copy()
            idx = np.random.randint(0, max(1, len(signal) - 50))
            corrupted[idx:idx+50] *= 0.1
            return corrupted

# Placeholder stubs for Days 28-29
def baseline_wander(signal, **kwargs):
    t = np.linspace(0, 1, len(signal))
    return signal + 0.5 * np.sin(2 * np.pi * 0.5 * t)

def emg_burst(signal, **kwargs):
    corrupted = signal.copy()
    idx = np.random.randint(0, max(1, len(signal) - 30))
    corrupted[idx:idx+30] += np.random.normal(0, 0.5, size=min(30, len(signal)-idx))
    return corrupted

def powerline_hum(signal, fs=250, **kwargs):
    t = np.arange(len(signal)) / fs
    return signal + 0.2 * np.sin(2 * np.pi * 50 * t)

def adc_quantization_noise(signal, bits=8, **kwargs):
    q_levels = 2 ** bits
    s_min, s_max = np.min(signal), np.max(signal)
    if s_max == s_min:
        return signal
    normalized = (signal - s_min) / (s_max - s_min)
    quantized = np.round(normalized * (q_levels - 1)) / (q_levels - 1)
    return quantized * (s_max - s_min) + s_min

# Aliases to prevent naming conflicts
quantization_noise = adc_quantization_noise


class PhysioShiftAugmentor:
    def __init__(self, p_augment=0.8, min_glitches=1, max_glitches=3):
        self.p_augment = p_augment
        self.min_glitches = min_glitches
        self.max_glitches = max_glitches
        self.glitch_map = {
            "motion_artifact_burst": motion_artifact_burst,
            "poor_contact_dropout": poor_contact_dropout,
            "baseline_wander": baseline_wander,
            "emg_burst": emg_burst,
            "powerline_hum": powerline_hum,
            "adc_quantization_noise": adc_quantization_noise,
        }

    def __call__(self, signal, fs=250):
        if random.random() > self.p_augment:
            return signal.copy(), []

        num_glitches = random.randint(self.min_glitches, self.max_glitches)
        chosen = random.sample(list(self.glitch_map.keys()), k=num_glitches)

        corrupted = signal.copy()
        for name in chosen:
            func = self.glitch_map[name]
            corrupted = func(corrupted, fs=fs)

        return corrupted, chosen


if __name__ == "__main__":
    print("Running PhysioShiftAugmentor test suite...")

    # Test baseline creation
    augmentor = PhysioShiftAugmentor(p_augment=1.0, min_glitches=1, max_glitches=3)
    clean_signal = np.sin(np.linspace(0, 10 * np.pi, 500))

    # Test augmentation execution
    corrupted, chosen = augmentor(clean_signal)
    assert len(chosen) >= 1 and len(chosen) <= 3, "Glitch selection count mismatch"
    assert corrupted.shape == clean_signal.shape, "Shape mismatch after augmentation"

    # Test pass-through execution
    clean_pass_augmentor = PhysioShiftAugmentor(p_augment=0.0)
    uncorrupted, chosen_none = clean_pass_augmentor(clean_signal)
    assert len(chosen_none) == 0, "Augmentations were applied when p_augment=0.0"
    np.testing.assert_array_equal(clean_signal, uncorrupted)

    print("All augmentor tests passed")