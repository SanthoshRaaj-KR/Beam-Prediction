# HANDOFF_PHASE_3 — CNN beam classifier + Run R0 (baseline)

Run on Kaggle GPU, 2026-10-14. `run_all.py` stage `baseline`, **39 s** for all
three seeds. **GATE PASSED.**

## 1. Model and training recipe

`models/beam_classifier.py::BeamCNN(n_beams=32)` and
`training/train_classifier.py::train_classifier(...)` — both reused
**unmodified** by Phase 7. Frozen hyperparameters:

```python
EPOCHS = 100
BATCH_SIZE = 64
LR = 1e-3            # Adam
N_VAL = 100          # 400/100 split within the 500 real samples
scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
# model selection: best validation Top-1 accuracy
```

Validation split is **stratified and real-only** in every run, by construction
(`_stratified_val_split`). The test set is evaluated **once**, after model
selection.

## 2. R0 results (real-only, 400 train / 100 val / 2000 test)

| Seed | Top-1 | Top-3 | Spectral eff. (%) | Best val Top-1 | Time |
|---|---|---|---|---|---|
| 0 | 0.7895 | 0.9390 | 85.44 | 0.840 | 17 s |
| 1 | 0.7670 | 0.9270 | 83.31 | 0.770 | 11 s |
| 2 | 0.7905 | 0.9290 | 85.25 | 0.800 | 11 s |

**Mean ± std (ddof=1), n=3:**

| Metric | Value |
|---|---|
| Top-1 | **0.7823 ± 0.0133** |
| Top-3 | **0.9317 ± 0.0064** |
| Spectral efficiency | **84.67 ± 1.18 %** |

Spectral efficiency is relative to the optimal-beam rate at a fixed **10 dB
SNR** (`evaluation/metrics.py`, `SNR_DB = 10.0`) — a reported assumption.

## 3. Versus the majority-class baseline

| | Top-1 |
|---|---|
| Majority class (class 16) | 0.5384 |
| R0 | 0.7823 ± 0.0133 |
| **Improvement** | **+24.39 pp** |

Top-3 exceeds the trivial baseline by +39.33 pp. The gate in `run_all.py`
required R0 to beat majority by ≥5 pp; it cleared that by a factor of ~5. The
classifier has learned real position/channel structure, not the class prior.

## 4. Wall-clock budget for Phase 7

R0 averaged **13 s/run** at 400 training samples. Cost scales roughly with
training-set size, so the 18 remaining runs (525→1400 samples) come to
**≈9 minutes** of GPU time, plus synthetic generation and labelling. The 21-run
design is cheap — there was never a reason to cut seeds.

## 5. Results CSV

`/kaggle/working/results/ablation_results.csv`, appended after **every** run.
Exact column order:

```
run_id, synthetic_ratio, seed, n_real, n_synth, n_total,
top1_acc, top3_acc, spectral_eff_pct, best_val_acc, epochs, timestamp
```

## 6. Overfitting observations (for the discussion)

Clear and expected overfitting — this *is* the data-scarcity regime the paper
is about:

- Training loss falls to **0.006–0.058** by epoch 100 (near-zero), while
  validation Top-1 plateaus at **0.69–0.84** and never improves past ~epoch 40.
- Validation accuracy is **non-monotonic** (seed 1: 0.770 → 0.690 → 0.750),
  which is why best-val-accuracy model selection matters rather than taking the
  final epoch.
- Test Top-1 (0.782) sits slightly *below* mean best-val (0.803), consistent
  with mild selection optimism on a 100-sample validation set.

This is the gap the synthetic data is meant to close, and it gives the paper a
concrete mechanism to discuss: with 400 real samples a 4.3M-parameter CNN
memorizes the training set within ~40 epochs.

## Note for Phase 7

The GAN training curve (stage `gan`) shows the Wasserstein estimate **falling
to ≈0.12 around iteration 700 and then rising to ≈0.62 by iteration 2,900** —
the critic pulling ahead of the generator. Expect a high discriminability score
in Phase 6 and treat it as a reportable finding, not a failure. See
`HANDOFF_PHASE_5.md`/`HANDOFF_PHASE_6.md` when written.
