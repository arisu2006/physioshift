
# PhysioShift

Repository for signal processing pipelines and model development.

---

## PyTorch Training Loop Mechanics

This section serves as a technical walkthrough for team members on the training mechanics implemented in `src/toy_training_loop.py`.

### Architectural Flow

```text
Dataset ---> DataLoader ---> Model (Forward Pass) ---> Loss Function ---> Backward Pass ---> Optimizer Step
```

### Key Components

#### 1. Custom Dataset (`torch.utils.data.Dataset`)
Handles dataset loading. Implements two mandatory protocols:
* `__len__()`: Returns total dataset size.
* `__getitem__(idx)`: Extracts a single feature-target tuple at index `idx`.

#### 2. DataLoader (`torch.utils.data.DataLoader`)
Wraps the dataset to provide batching, dataset shuffling, and parallel execution.

#### 3. Neural Network (`torch.nn.Module`)
Defines the graph layers inside `__init__()` and controls data propagation through `forward(x)`.

#### 4. The 5-Step Training Batch Cycle

For each iteration over mini-batches in an epoch:

1. **`optimizer.zero_grad()`**: Resets gradients stored from the previous iteration.
2. **`outputs = model(inputs)`**: Executes the forward pass to compute outputs.
3. **`loss = criterion(outputs, targets)`**: Calculates the discrepancy between prediction and target.
4. **`loss.backward()`**: Triggers Autograd to compute gradients across all network parameters.
5. **`optimizer.step()`**: Adjusts network weights according to computed gradients and the optimization rule.

---

### Verification Run

Execute the script to verify pipeline mechanics:

```bash
python src/toy_training_loop.py
```
\# PhysioShift — Physiological Dataset Verification



This repository contains the initial data verification workflow for multi-modal physiological datasets (ECG, PPG, and accelerometer signals).



All verification code, signal shape checks, sampling rate inspections, and metadata extractions are implemented in \[`notebooks/00\_dataset\_verification.ipynb`](./notebooks/00\_dataset\_verification.ipynb).



\---



\## 1. Summary of Open Datasets Verified



| Dataset | Primary Format | Access / Loader Method | Sampling Rate ($f\_s$) | Key Channels / Signals | Verification Status |

| :--- | :--- | :--- | :--- | :--- | :---: |

| \*\*PTB-XL\*\* | PhysioNet WFDB (`.hea` / `.dat`) | `wfdb.rdrecord()` | 100 Hz / 500 Hz | 12-lead ECG (I, II, III, aVR, aVL, aVF, V1–V6) | ✅ \*\*Verified\*\* |

| \*\*MIT-BIH Arrhythmia\*\* | PhysioNet WFDB (`.hea` / `.dat` / `.atr`) | `wfdb.rdrecord()`, `wfdb.rdann()` | 360 Hz | Modified Lead II (MLII), V1 / V2 / V4 / V5 | ✅ \*\*Verified\*\* |

| \*\*WESAD\*\* | UCI Pickle (`.pkl`) | `pickle.load(f, encoding='latin1')` | 700 Hz (Chest) / 64 Hz (Wrist) | ECG, EDA, EMG, Temp (Chest) / BVP/PPG, EDA, TEMP (Wrist) | ✅ \*\*Verified\*\* |

| \*\*PPG-DaLiA\*\* | UCI Pickle (`.pkl`) | `pickle.load(f, encoding='latin1')` | 700 Hz (Chest) / 64 Hz (Wrist) | ECG, ACC (Chest) / BVP/PPG, ACC, TEMP (Wrist) | ✅ \*\*Verified\*\* |

| \*\*Multi-site PPG\*\* | Tabular / Binary (`.csv` / `.pkl` / `.hdf5`) | `pandas.read\_csv()` / `pickle.load()` | Variable (Site-dependent) | Multi-site PPG (Earlobe, Finger, Toe) | ✅ \*\*Verified\*\* |



\---



\## 2. Technical Breakdown \& Data Specifications



\### 2.1 PTB-XL (PhysioNet WFDB Format)

\* \*\*Overview:\*\* A large 12-lead ECG dataset containing 21,837 clinical ECG records from 18,885 patients.

\* \*\*Storage Standard:\*\* Standard WFDB format consisting of paired plain-text header files (`.hea`) and binary signal files (`.dat`).

\* \*\*Loading Mechanism:\*\*

&#x20; ```python

&#x20; import wfdb

&#x20; record = wfdb.rdrecord('data/ptb-xl/records100/00000/00001\_lr')

&#x20; signal = record.p\_signal  # Shape: (1000, 12) at 100 Hz
Clean baseline (26 Aug): F1 = 0.4663, ECE = 0.2292
Augmented baseline (2 Sep): F1 = 0.4663, ECE = 0.0869

ECE improved significantly from 0.23 to 0.09, showing that data augmentation effectively mitigates baseline overconfidence while keeping Split C macro-F1 steady at 0.47. Augmentation should definitely be carried into the SSL phase.