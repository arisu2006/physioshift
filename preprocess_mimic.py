import os
import glob
import numpy as np
import wfdb
from src.harmonize.dataset import PhysioShiftDataset

def process_mimic():
    pipeline = PhysioShiftDataset(target_fs=100.0, window_sec=10.0, overlap=0.5)
    
    hea_files = glob.glob("data/raw/*.hea")
    if not hea_files:
        print("No .hea files found in data/raw!")
        return

    all_windows = []
    all_labels = []

    print(f"Found {len(hea_files)} record(s). Processing...")

    for hea_path in hea_files:
        record_base = os.path.splitext(hea_path)[0]
        try:
            record = wfdb.rdrecord(record_base)
            signal = record.p_signal[:, 0]  # Take Lead I / first ECG channel
            fs = float(record.fs)

            # Process: resample -> bandpass filter -> window
            windows = pipeline.process_signal(signal, original_fs=fs, modality="ecg", domain_id="D1_MIMIC_clinical_chest")
            
            all_windows.append(windows)
            all_labels.append(np.zeros(len(windows), dtype=int))
            print(f"Processed {record_base}: generated {windows.shape[0]} windows with shape {windows.shape}")
        except Exception as e:
            print(f"Error processing {record_base}: {e}")

    if not all_windows:
        print("No windows generated.")
        return

    x_harmonized = np.concatenate(all_windows, axis=0)
    y_harmonized = np.concatenate(all_labels, axis=0)

    os.makedirs("data/processed", exist_ok=True)
    np.save("data/processed/x_harmonized.npy", x_harmonized)
    np.save("data/processed/y_harmonized.npy", y_harmonized)

    print("-" * 50)
    print("Preprocess finished successfully!")
    print(f"Saved x_harmonized shape: {x_harmonized.shape}")
    print(f"Saved y_harmonized shape: {y_harmonized.shape}")

if __name__ == "__main__":
    process_mimic()