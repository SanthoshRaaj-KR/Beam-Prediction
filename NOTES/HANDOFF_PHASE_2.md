# HANDOFF_PHASE_2 — Beam codebook & ground-truth labels

Run on Kaggle, 2026-10-14. `run_all.py` stage `labels`, **1 s**.

## 1. Codebook code used

Hand-rolled DFT, **not** `deepmimo.steering_vec()` (see guide §0.5 — that
returns one array-response vector for one direction; 32 beams would need 32
calls with hand-mapped angles. Closed, do not revisit).

```python
def build_codebook(n_ant=64, n_beams=32):
    n = np.arange(n_ant).reshape(-1, 1)
    k = np.arange(n_beams).reshape(1, -1)
    W = np.exp(1j * 2 * np.pi * n * k / n_beams) / np.sqrt(n_ant)
    return torch.from_numpy(W.T.copy())        # [32, 64] complex, row k = beam k
```

Labelling: `argmax_k mean_s |w_k^H h_s|^2` — the subcarrier-averaged
beamforming gain, computed per sample.

## 2. Label histogram

- **32 of 32 classes populated.** No degeneracy.
- Plot: `/kaggle/working/results/plots/fig_label_hist.png` → **Figure 1**.
- Per-class counts: `label_counts` in `/kaggle/working/state.json`.

## 3. Majority-class baseline

```
0.5384   (class 16)
```

**This is high and must be handled honestly in the paper.** 53.84% of the 2,500
labelled users share a single optimal beam. The cause is geometric, not a bug:
per `HANDOFF_PHASE_1.md`, the user region is a 550 m × 36 m strip, so most users
sit in a narrow angular sector from the BS and fall into the same DFT beam.
Class 16 is the mid-codebook beam.

Consequences to carry into the writing:

- Report the majority-class baseline prominently; a 32-class problem whose
  trivial baseline is 53.8% is not as hard as "32 classes" suggests, and a
  reviewer will notice if it is buried.
- Lean on **Top-3 and spectral efficiency** alongside Top-1 — both are less
  sensitive to the imbalance.
- The limitations section should state that a single BS / single RX strip yields
  an imbalanced beam distribution, and that a multi-BS or wider-region study
  would be a natural extension.

## 4. Classes populated

32 / 32.

## 5. Normalized vs. unnormalized labels

The labeller is **provably invariant** to the per-sample Frobenius scaling:
dividing `h_s` by a positive constant `c` scales every `|w_k^H h_s|^2` by
`1/c²`, which cannot change the `argmax` over `k`.

Verified empirically in `smoke_test.py` (label scale-invariance assertion).
Labels for this run were computed on the **normalized** tensors, which is
therefore equivalent to computing them on the raw channels.

## 6. Saved artifacts

| File | Shape | dtype |
|---|---|---|
| `/kaggle/working/data/codebook.pt` | `[32, 64]` | complex128 |
| `/kaggle/working/data/seed_labels.pt` | `[500]` | `torch.int64` |
| `/kaggle/working/data/test_labels.pt` | `[2000]` | `torch.int64` |

Labels are integers in `[0, 31]`. The same labeller is the **only** source of
beam labels in the project — it labels synthetic samples in Phase 7 too. The
GAN never produces labels.
