# PhysioShift — Baseline Supervised Ablation Table

**Model:** LightweightResNet1D (500K params, no SSL)
**Label:** Heart-rate bucket (low <60bpm / normal 60-100bpm / high >100bpm), derived via peak detection — a shared physiological label valid across all domains, since domain identity cannot serve as a label for Split B/C (train/test domains are disjoint by construction).
**Training config:** 5 epochs, batch_size=16, lr=1e-3, Adam, CrossEntropyLoss

| Row | Model | Split | Description | Train windows | Test windows | F1-macro | ECE |
|---|---|---|---|---|---|---|---|
| 1 | Baseline supervised | A | In-device (easy) | 378 | 94 | 0.8278 | 0.0138 |
| 1 | Baseline supervised | B | Unseen device D5 (moderate) | 458 | 14 | 0.0667 | 0.9224 |
| 1 | Baseline supervised | C | Clinical→consumer (hardest) | 361 | 111 | 0.4663 | 0.2292 |

## Key finding

Performance degrades sharply with increasing domain shift: F1 drops from
0.83 (in-device) to 0.47 (clinical→consumer) to 0.07 (unseen device).
Split B is also badly miscalibrated (ECE=0.92) — the model is
overconfident on data far outside its training distribution. This
motivates the SSL/domain-adaptation work planned in later phases.

## Checkpoints and diagrams

- `results/checkpoints/split_a_baseline.pt`
- `results/checkpoints/split_b_baseline.pt`
- `results/checkpoints/split_c_baseline.pt`
- `results/reliability_diagrams/split_a_reliability.png`
- `results/reliability_diagrams/split_b_reliability.png`
- `results/reliability_diagrams/split_c_reliability.png`