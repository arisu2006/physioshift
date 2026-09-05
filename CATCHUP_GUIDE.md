# PhysioShift — Catch-Up Guide: 21, 24, 27, 30 Aug (Member 3)

All four of these dates belong to you (Member 3) in the Master Execution
Calendar. None of them assign a specific research paper to read (that only
happens in Phase 1 — PaPaGei/CLOCS/M2AE/DANN) — each day pairs a **concept
to learn** with a **thing to build**. Below is both, in order, with working
code for each. Run them in this sequence since 24 Aug depends on 21 Aug's
model, and 30 Aug depends on 27 Aug's functions.

---

## 21 Aug — Baseline Supervised Training Loop
**File:** `src/train/train_baseline.py`

### Concept
- **Cross-entropy loss**: standard multi-class classification loss; penalizes confident wrong predictions heavily.
- **Class-imbalance handling**: PhysioShift's classes (arrhythmia types etc.) won't be evenly distributed. Fix: pass `class_weight='balanced'`-style weights into `nn.CrossEntropyLoss(weight=...)` so rare classes contribute more to the loss.
- **F1-macro**: `sklearn.metrics.f1_score(y_true, y_pred, average='macro')` — averages per-class F1 equally, so it doesn't get inflated by the majority class the way accuracy does.
- **Early stopping**: stop training once validation F1 stops improving for `patience` epochs, and roll back to the best checkpoint — prevents overfitting on a small dataset.

### Steps
1. `cd` into your repo root (or use the `physioshift/` folder in this delivery as a starting skeleton).
2. Run the smoke test first, exactly as the calendar specifies — "smoke-test on a tiny data subset":
   ```
   python src/train/train_baseline.py --smoke-test --epochs 10 --no-wandb
   ```
3. Once that runs clean, install and log in to W&B, then drop `--no-wandb`:
   ```
   pip install wandb
   wandb login
   python src/train/train_baseline.py --smoke-test --epochs 20
   ```
4. Swap `make_smoke_test_data()` for your real harmonized-cache loader (from `src/harmonize/dataset.py`) once you're pointing at real data instead of the fake windows.
5. Swap the stub `LightweightResNet1D` in `src/models/resnet1d.py` for the team's real Day-20 model file if a teammate has already pushed one — the interface (`forward(x) -> [batch, num_classes]`) is what matters, not this exact implementation.
6. **Deliverable to push:** `src/train/train_baseline.py` + a W&B run link showing a completed smoke-test curve (screenshot the run page).

---

## 24 Aug — Expected Calibration Error (ECE) + Reliability Diagram
**File:** `src/eval/calibration.py`

### Concept
- **ECE** buckets predictions by confidence, then measures the gap between confidence and actual accuracy in each bucket, weighted by bucket size. Near 0 = well-calibrated.
- **Reliability diagram**: bar chart of accuracy vs. confidence per bucket, with a diagonal reference line for perfect calibration. Bars below the line = overconfidence (very common for neural nets, and expected here).

### Steps
1. Run the standalone smoke test (uses a deliberately overconfident fake model so you can see the failure pattern immediately):
   ```
   python src/eval/calibration.py --smoke-test
   ```
2. This produces `reliability_diagram.png` — open it and confirm the bars sit below the diagonal (overconfident), which is the expected result.
3. Wire it to your real Day-21 baseline: after training, run the baseline on your held-out Split C set, collect softmax confidences + predicted classes + true labels, then call:
   ```python
   from src.eval.calibration import compute_ece, plot_reliability_diagram
   ece, bin_stats = compute_ece(confidences, predictions, labels, n_bins=15)
   plot_reliability_diagram(bin_stats, ece, save_path="split_c_reliability.png")
   ```
4. **Deliverable to push:** `src/eval/calibration.py` + the real reliability diagram from your Split C baseline. Per the calendar's own note, document explicitly in `RESULTS.md` if it comes out overconfident (it almost certainly will) — that's a real, expected finding, not a bug.

---

## 27 Aug — Motion Artifact Burst & Poor-Contact Dropout
**File:** `src/augment/motion_and_contact.py`

### Concept (the "why", which the calendar explicitly asks you to understand, not just code)
- **Motion artifact burst**: a wearable shifting on skin (arm swing, walking) causes a short (~0.3–2s), high-amplitude, broadband disturbance that swamps the true signal, then settles back.
- **Poor-contact dropout**: loose strap / sweat / movement raises electrode-skin (ECG) or optode-skin (PPG) impedance, which *attenuates* the signal toward the sensor's floor for a sustained period — it suppresses the signal rather than adding noise on top of it. That's the key conceptual difference from the burst.

### Steps
1. Run it directly — it generates before/after plots with hardware-cause captions built in, exactly as the deliverable spec asks:
   ```
   python src/augment/motion_and_contact.py
   ```
2. Inspect `motion_artifact_burst.png` and `poor_contact_dropout.png` — confirm the burst looks like a short violent spike and the dropout looks like sustained amplitude collapse, not the same kind of distortion.
3. Once you have real harmonized signals (from Phase 2's pipeline), swap `make_synthetic_signal()` for a real windowed sample from one of your 5 datasets, per dataset/modality (ECG vs PPG have different plausible burst/dropout durations — tune `burst_duration_range` / `dropout_duration_range` if needed).
4. **Deliverable to push:** `src/augment/motion_and_contact.py` + the two before/after plots with captions (these double as paper figure captions later, so keep the one-line hardware-cause wording).

---

## 30 Aug — PhysioShiftAugmentor (Full Composition Class)
**File:** `src/augment/augmentor.py`

### Concept
- Real wear conditions rarely produce exactly one artifact type in isolation — composing 1–3 corruptions per training example better matches deployment than a single fixed corruption.
- `p` (probability that *any* augmentation is applied at all) is deliberately exposed as a parameter now, since the 03 Sep task turns it into a config-driven ablation knob (0.3 / 0.5 / 0.7).

### Steps
1. This day's task **depends on all 6 corruption functions existing** — 2 are yours (27 Aug), 4 belong to Member 1 (28 Aug: `adc_quantization_noise`, `gsr_electrode_saturation`) and Member 2 (29 Aug: `ppg_optical_interference`, `baseline_wander`). Since you're catching up solo, I included working stand-ins for those four in `src/augment/other_corruptions_stub.py` so the class is testable today — **replace that file's imports with your teammates' real versions once they're pushed** (same function signatures, so it's a one-line swap in `augmentor.py`).
2. Run the built-in unit tests — one per corruption function, plus a composition test, exactly matching the deliverable spec:
   ```
   python -m src.augment.augmentor
   ```
3. Use it in your training pipeline like a standard on-the-fly transform:
   ```python
   from src.augment.augmentor import PhysioShiftAugmentor
   augmentor = PhysioShiftAugmentor(p=0.5, min_corruptions=1, max_corruptions=3, fs=100)
   corrupted_signal = augmentor(clean_signal)
   ```
4. **Deliverable to push:** `src/augment/augmentor.py`, fully documented and unit-tested (the file already has one test per corruption + one composition test, per spec).

---

## Order to actually do this in, given you're behind
1. **21 Aug** first (nothing else depends on anything, and 24 Aug needs its output).
2. **24 Aug** next (needs a trained baseline from 21 Aug to be meaningful, though the smoke test runs standalone).
3. **27 Aug** — independent of the above two, can run in parallel/before if you want a break from training loops.
4. **30 Aug** last (depends on 27 Aug's two functions, and on stub-or-real versions of the other four).

Push each day's deliverable as its own commit/tag if you're behind on GitHub too — matches the calendar's "tagged release + updated RESULTS.md" expectation per phase, and gives you a clean paper trail showing the work actually happened even if the dates slipped.
