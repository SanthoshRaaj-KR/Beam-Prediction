# Implementation Phases — GAN-Augmented mmWave Beamforming

> **READ THIS FIRST — how this document is meant to be used.**
>
> This project is executed **one phase at a time, each by a different AI assistant with no memory of the previous phases.** Every phase below is therefore written to be **self-contained**: it restates the context needed, gives exact architectures and code, and ends with a **handoff artifact** that the next phase depends on.
>
> **The rule that makes this work:** at the end of every phase, the executing AI must write a file called `HANDOFF_PHASE_<N>.md` containing the specified contents. When you start the next phase, you paste that handoff file (plus anything else the phase's "What to paste" block lists) alongside the phase instructions. Without the handoff, the next AI is blind.
>
> **Companion documents:**
> - `beamforming_project_spec.md` — original research spec (background/motivation).
> - `execution_and_paper_guide.md` — the overall plan and rationale, including §0.5 Implementation Notes & Errata (real API gotchas discovered during execution — keep updating it).
> - `PAPER_DRAFT.md` — the paper skeleton to fill in at Phase 9.

---

## Project in one paragraph (paste this into EVERY phase)

> **PROJECT CONTEXT BLOCK — copy this verbatim at the start of every phase prompt.**
>
> We are training a conditional WGAN-GP to synthesize millimeter-wave Channel State Information (CSI) matrices conditioned on user position, using a deliberately small "real" seed set (500 samples) from the DeepMIMO `o1_28` ray-tracing dataset. We then train a small CNN beam classifier (predicts the best beamforming codebook index out of 32 from a CSI input) on mixtures of real + GAN-synthetic data at seven synthetic-to-real ratios (0%, 25%, 50%, 75%, 100%, 150%, 200%), each repeated with 3 random seeds, and evaluate every run on the same held-out 2000-sample real test set. The output is an ablation curve of beam-prediction accuracy versus synthetic data ratio. This is an undergraduate research project targeting a regional/student IEEE conference. Priorities: correctness, reproducibility, and finishing — not state-of-the-art performance. Compute is Kaggle free-tier GPU (T4), 7-hour session limit.
>
> Data shapes used throughout: CSI tensor `[N, 2, 64, 32]` (2 = real/imag, 64 = antennas, 32 = subcarriers), position `[N, 2]` (normalized x,y), beam label `[N]` (integer in `[0,31]`).

---

## Global conventions (all phases must obey)

1. **Seeds:** every script sets `torch.manual_seed(SEED)`, `np.random.seed(SEED)`, `random.seed(SEED)`. Default `SEED=42` unless the phase says otherwise.
2. **Paths:**
   - Big/raw/scratch data → `/root/deepmimo_scenarios/` (root filesystem, ~1.1 TB free, wiped at session end).
   - Small artifacts to keep (tensors, checkpoints, CSVs, plots) → `/kaggle/working/` (**20 GB hard cap** — never extract raw scenarios here).
   - Persist across sessions by saving `/kaggle/working/` outputs as a **Kaggle Dataset**, or committing the notebook version.
3. **Checkpointing:** any training loop longer than ~15 minutes saves `state_dict()`s every N iterations and resumes if a checkpoint exists.
4. **Never tune hyperparameters per-ablation-run.** The classifier's architecture, LR, scheduler, epochs, and batch size are frozen after Phase 3 and identical across all 21 runs in Phase 7. Only the training data changes. This is what makes the ablation valid.
5. **Log everything to CSV** as you go — not at the end.
6. **If reality contradicts this document, reality wins** — record the discrepancy in `execution_and_paper_guide.md` §0.5 and in your handoff file.

---

# PHASE 0 — Environment & Scenario Acquisition

### What to paste alongside this phase
- The PROJECT CONTEXT BLOCK above.
- Nothing else (this is the first phase).

### Known facts (already established — do not re-derive)
- `pip install DeepMIMO` installs version **4.0.3**; the importable module is **lowercase `deepmimo`**.
- **You must restart the kernel** after install or the import fails with `ModuleNotFoundError`.
- `deepmimo.get_available_scenarios()` lists only *locally downloaded* scenarios — it returns `[]` on a fresh session. Use `deepmimo.search({'bands': ['mmW']})` to query the online catalog.
- The scenario we want is named **`o1_28`** (lowercase, band-suffixed). Not `O1`.
- `deepmimo.download('o1_28')` defaults to `/kaggle/working/` and **will fail** with `OSError: [Errno 28] No space left on device` — the 2.28 GB zip expands past the 20 GB `/kaggle/working` cap. Download to the root filesystem instead.
- DeepMIMO's download endpoint **rate-limits with HTTP 429** on repeated requests. If you hit it, wait 10–15 minutes; do not retry in a loop.
- If a download succeeds but extraction fails, **delete only the extraction folder, not the zip** — re-extract locally rather than re-downloading.

### Steps

```python
# Cell 1
!pip install DeepMIMO
# --> RESTART KERNEL NOW (Run > Restart Session) before continuing

# Cell 2
import deepmimo, os, torch, numpy as np
print(deepmimo.__version__)          # expect 4.0.3
print(torch.cuda.is_available())      # expect True
!df -h /                              # confirm root fs has hundreds of GB free
!df -h /kaggle/working                # 20G cap — do not fill this

# Cell 3 — download raw scenario OUTSIDE /kaggle/working
os.makedirs('/root/deepmimo_scenarios', exist_ok=True)
deepmimo.download('o1_28', output_dir='/root/deepmimo_scenarios')
```

### Then map the API (the next phase depends on this)
Run and capture the **full output** of:

```python
help(deepmimo.load)
help(deepmimo.generate)
help(deepmimo.ChannelParameters)
print([x for x in dir(deepmimo) if not x.startswith('_')])
```

Then load the scenario and inspect the object:

```python
dataset = deepmimo.load('o1_28')     # adjust to the real signature from help()
print(type(dataset))
print(dir(dataset))
# Find: channel matrices, user positions, number of users, tx/rx set structure
```

### Checkpoint (must all be true before Phase 1)
- [ ] `deepmimo.__version__ == '4.0.3'`, GPU available.
- [ ] `o1_28` downloaded and extracted successfully outside `/kaggle/working`.
- [ ] A `Dataset` (or equivalent) object loads without error.
- [ ] You can print the number of users and the shape/dtype of at least one channel matrix.

### HANDOFF_PHASE_0.md must contain
1. The exact working install/import/download code that succeeded.
2. Full `help()` text for `deepmimo.load`, `deepmimo.generate`, `deepmimo.ChannelParameters`.
3. The exact path where `o1_28` was extracted.
4. `type(dataset)`, `dir(dataset)`, and the structure of the loaded object — specifically **where the channel matrix lives, where user positions live, and how many users exist**.
5. Shape and dtype of one raw channel matrix, printed.
6. Any errors hit and how they were resolved.

---

# PHASE 1 — Data Extraction & Preprocessing

### What to paste alongside this phase
- PROJECT CONTEXT BLOCK.
- **`HANDOFF_PHASE_0.md` in full** (the API signatures are essential — do not guess them).

### Goal
Turn the raw DeepMIMO scenario into six small tensors saved to `/kaggle/working/data/`:

| File | Shape | Notes |
|---|---|---|
| `seed_csi.pt` | `[500, 2, 64, 32]` | float32, normalized |
| `seed_pos.pt` | `[500, 2]` | float32, normalized to [-1,1] |
| `seed_norm.pt` | `[500]` | original Frobenius norms (for denormalization later) |
| `test_csi.pt` | `[2000, 2, 64, 32]` | float32, normalized |
| `test_pos.pt` | `[2000, 2]` | float32 |
| `test_norm.pt` | `[2000]` | original Frobenius norms |

### Channel configuration
Use `ChannelParameters` (exact field names come from Phase 0's handoff) to set:
- Carrier frequency: **28 GHz** (should be intrinsic to `o1_28`).
- BS antenna array: **64-element ULA** (e.g. `[64, 1]` or `[1, 64, 1]` depending on the API's panel convention).
- UE antenna: **1 element** (single-antenna receivers — keeps `H` as `[64, 32]`).
- OFDM subcarriers: **32**. If `o1_28` does not support exactly 32, use the nearest supported value, **document the actual number**, and propagate the change everywhere (all `32`s in later phases become that number).

### Preprocessing steps

```python
import torch, numpy as np

# 1. Pull complex channel H per user -> shape [64, 32] complex
#    and position (x, y, z) per user. (API per Phase 0 handoff.)

# 2. Drop z (O1 is ground-level; elevation is ~constant).

# 3. FILTER OUT dead users. O1 contains users with no valid path
#    (blocked / out of coverage) whose channels are all-zero or NaN.
#    Keep only users where np.isfinite(H).all() and ||H||_F > 0.
#    Report how many users were dropped — this goes in the paper.

# 4. Real/imag split -> [2, 64, 32]
csi = np.stack([H.real, H.imag], axis=0).astype(np.float32)

# 5. Per-sample Frobenius normalization; STORE the norm.
fro = np.linalg.norm(csi)          # over the whole [2,64,32] tensor
csi_normalized = csi / fro

# 6. Positions -> normalize each axis to [-1, 1] using the min/max
#    over the FULL retained user set (record these bounds; the GAN
#    conditioning at Phase 5/7 must use the identical transform).

# 7. Split: sample 2500 users at random WITHOUT replacement (seed 42),
#    first 500 -> seed set, next 2000 -> test set.
#    CRITICAL: these must be disjoint. Assert it.
```

**Split integrity requirement:** the test set must be *disjoint in user index* from the seed set. Assert `len(set(seed_idx) & set(test_idx)) == 0` and save both index arrays (`seed_idx.npy`, `test_idx.npy`) so the split is reproducible.

> **Optional, worth 10 minutes:** rather than a purely random split, consider a **spatially disjoint** split (e.g. seed users from one region of the grid, test users from another). This is a *harder, more honest* generalization test and reviewers like it. If you do this, document it; if you don't, random split is acceptable — just don't claim spatial generalization in the paper.

### Checkpoint
- [ ] All six `.pt` files saved and reload correctly with the shapes in the table.
- [ ] `assert seed_csi.shape == (500, 2, 64, 32)` passes.
- [ ] No NaNs/Infs: `torch.isfinite(seed_csi).all()` is `True`.
- [ ] Seed/test index sets are disjoint (asserted).
- [ ] Position values lie within `[-1, 1]`.

### HANDOFF_PHASE_1.md must contain
1. The exact `ChannelParameters` configuration used (all fields, verbatim code).
2. **The actual subcarrier count used** (32 or otherwise) and antenna count.
3. Number of users in the raw scenario, number dropped as invalid, number retained.
4. The position normalization bounds (min/max per axis) — later phases need these to condition the GAN identically.
5. The normalization convention chosen (per-sample Frobenius) confirmed, and confirmation that norms were saved.
6. File paths of all six tensors + the two index arrays.
7. Printed `.shape` and `.dtype` of every saved tensor.

---

# PHASE 2 — Beam Codebook & Ground-Truth Labels

### What to paste alongside this phase
- PROJECT CONTEXT BLOCK.
- `HANDOFF_PHASE_1.md` (need the real antenna/subcarrier counts and file paths).

### Goal
Produce `seed_labels.pt` `[500]` and `test_labels.pt` `[2000]`, integers in `[0, 31]`, plus a saved codebook `codebook.pt` `[32, 64]` complex.

### Codebook definition (DFT, 32 beams, 64-element ULA)

```python
import numpy as np, torch

N_ANT, N_BEAMS = 64, 32
n = np.arange(N_ANT).reshape(-1, 1)          # [64,1]
k = np.arange(N_BEAMS).reshape(1, -1)        # [1,32]
# Standard DFT beam steering matrix, columns = beams
W = np.exp(1j * 2 * np.pi * n * k / N_BEAMS) / np.sqrt(N_ANT)   # [64,32]
codebook = torch.from_numpy(W.T)             # [32,64], row k = beam k
```

> **Check first:** Phase 0 revealed DeepMIMO exposes `deepmimo.steering_vec()`. Run `help(deepmimo.steering_vec)` — if it produces an equivalent ULA steering matrix, prefer it (fewer hand-rolled-math bugs) and note the substitution in the handoff. Either path is fine; just document which was used.

### Label computation

```python
# H_complex: [64, 32] complex (reconstruct from the [2,64,32] real tensor:
#            H = csi[0] + 1j*csi[1])
# Beamforming gain for beam k on subcarrier s:  |w_k^H h_s|^2
# Average over subcarriers, argmax over beams.

def optimal_beam(H_complex, codebook):          # H:[64,S], codebook:[32,64]
    gains = np.abs(codebook.conj() @ H_complex) ** 2    # [32, S]
    return int(gains.mean(axis=1).argmax())
```

**Important:** compute labels on the **normalized** CSI — per-sample Frobenius scaling does not change the argmax, so labels are identical either way. (Verify this on 10 samples rather than assuming it.)

### Sanity checks (do not skip — a degenerate label distribution invalidates everything downstream)
- [ ] Plot a histogram of labels across the 2500 samples.
- [ ] Confirm **more than ~10 of the 32 classes are populated**. If 90%+ of samples land on 1–2 beams, something is wrong (wrong array axis, wrong conjugation, or the users are all clustered in one direction) — debug before proceeding.
- [ ] Report the majority-class frequency. **This is your trivial baseline** — a classifier that always predicts the most common beam achieves this accuracy. Every result in Phase 7 must beat it, and the paper must report it.

### Checkpoint
- [ ] `seed_labels.pt`, `test_labels.pt`, `codebook.pt` saved.
- [ ] Label histogram plotted and saved to `/kaggle/working/results/plots/label_hist.png`.
- [ ] Majority-class baseline accuracy recorded.

### HANDOFF_PHASE_2.md must contain
1. The exact codebook code used (hand-rolled or `steering_vec`).
2. Label histogram: counts per class, and the plot path.
3. **Majority-class baseline accuracy** (a single number, e.g. "0.14") — the paper needs this.
4. Number of distinct beam classes actually populated.
5. Confirmation that labels are identical on normalized vs. unnormalized CSI.
6. File paths for all saved artifacts.

---

# PHASE 3 — CNN Beam Classifier + Run R0 (baseline)

### What to paste alongside this phase
- PROJECT CONTEXT BLOCK.
- `HANDOFF_PHASE_1.md` and `HANDOFF_PHASE_2.md`.

### Goal
A reusable, frozen training function `train_classifier(train_csi, train_labels, seed) -> metrics`, and the **R0** result (real data only, 0% synthetic) at 3 seeds.

### Exact architecture (freeze this — all 21 runs use it unchanged)

```python
import torch.nn as nn

class BeamCNN(nn.Module):
    def __init__(self, n_beams=32):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(2, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            # [B,32,64,32]
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            # [B,64,32,16]
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),
            # [B,128,16,8]
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),                       # [B, 16384]
            nn.Linear(128*16*8, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, n_beams),            # logits, NO softmax
        )
    def forward(self, x):
        return self.classifier(self.features(x))
```
~4.3 M parameters, dominated by the first `Linear`. Trains in seconds/epoch on a T4.

### Frozen training recipe
| Setting | Value |
|---|---|
| Loss | `nn.CrossEntropyLoss()` |
| Optimizer | `Adam(lr=1e-3)` |
| Scheduler | `CosineAnnealingLR(T_max=epochs)` |
| Epochs | 100 |
| Batch size | 64 |
| Train/val split | 400/100 within the 500 real samples (stratified if possible) |
| Model selection | keep weights at **best validation accuracy**, then evaluate once on test |
| Device | CUDA |

**Rules:**
- The validation set is drawn **only from real data**, never synthetic, in every run — otherwise model selection is contaminated by GAN artifacts.
- The 2000-sample test set is touched **exactly once per run**, at the end.
- With 500 samples this will overfit; that is expected and is the entire point of the study.

### Metrics to return (implement in `evaluation/metrics.py`)
1. **Top-1 accuracy** — `(pred.argmax(1) == y).float().mean()`
2. **Top-3 accuracy** — ground truth within top-3 logits.
3. **Spectral efficiency ratio** — deferred to Phase 8 if it slows you down, but implement the hook now:
   ```
   SE_pred = log2(1 + SNR * gain_pred),  gain_pred = |w_pred^H h|^2 averaged over subcarriers
   SE_opt  = log2(1 + SNR * gain_opt)
   report mean(SE_pred / SE_opt) * 100%
   ```
   Fix `SNR = 10 dB` (state this value in the paper; it is a reported assumption, not a tuned parameter). Use the stored Frobenius norms to restore true scale before computing gains.

### Run R0
Train on the 500 real samples only, at seeds `[0, 1, 2]`. Append one CSV row per run to `/kaggle/working/results/ablation_results.csv` with columns:

```
run_id, synthetic_ratio, seed, n_real, n_synth, n_total,
top1_acc, top3_acc, spectral_eff_pct, best_val_acc, epochs, timestamp
```

### Checkpoint — this is the single most important gate in the project
- [ ] End-to-end pipeline works: load tensors → train → evaluate on held-out test set.
- [ ] R0 recorded at 3 seeds with mean ± std.
- [ ] **R0 Top-1 accuracy clearly beats the majority-class baseline from Phase 2.** If it does not, stop and debug — the labels, the data, or the training loop is broken, and no amount of GAN work will fix it.
- [ ] The training function is parameterized so Phase 7 can call it with different data mixes without editing it.

### HANDOFF_PHASE_3.md must contain
1. The full, final `BeamCNN` code and the complete training function (verbatim — Phase 7 reuses it unmodified).
2. R0 results: Top-1, Top-3, spectral efficiency, per seed and as mean ± std.
3. Comparison against the majority-class baseline.
4. Wall-clock training time per run (Phase 7 needs this to budget 18 more runs).
5. Path to `ablation_results.csv` and its exact column order.
6. Any overfitting observations (train vs. val curves) — useful for the paper's discussion.

---

# PHASE 4 — WGAN-GP Generator & Critic (implementation only)

### What to paste alongside this phase
- PROJECT CONTEXT BLOCK.
- `HANDOFF_PHASE_1.md` (data shapes/paths, position normalization bounds).

### Goal
Working `Generator`, `Critic`, and a gradient-penalty function, all shape-verified with dummy forward passes. **No real training in this phase** — this is deliberate, to isolate shape bugs from training bugs.

### Generator (exact — this shape math has been verified)

```python
class Generator(nn.Module):
    def __init__(self, z_dim=100, cond_dim=2):
        super().__init__()
        self.fc = nn.Linear(z_dim + cond_dim, 256*4*2)      # -> [B,256,4,2]
        self.net = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(),  # [B,128,8,4]
            nn.ConvTranspose2d(128,  64, 4, 2, 1), nn.BatchNorm2d(64),  nn.ReLU(),  # [B,64,16,8]
            nn.ConvTranspose2d(64,   32, 4, 2, 1), nn.BatchNorm2d(32),  nn.ReLU(),  # [B,32,32,16]
            nn.ConvTranspose2d(32,   16, 4, 2, 1), nn.BatchNorm2d(16),  nn.ReLU(),  # [B,16,64,32]
            nn.Conv2d(16, 2, 3, 1, 1), nn.Tanh(),                                    # [B,2,64,32]
        )
    def forward(self, z, c):
        x = self.fc(torch.cat([z, c], dim=1)).view(-1, 256, 4, 2)
        return self.net(x)
```

**Why a `4×2` seed and four upsampling stages:** the target is `64×32` (a 2:1 aspect ratio). Starting at `4×2` and doubling four times gives `4→8→16→32→64` on the antenna axis and `2→4→8→16→32` on the subcarrier axis — landing exactly on `[64,32]` with clean `stride=2, padding=1` arithmetic and **no cropping or padding hacks**. (An earlier draft of the spec started from `4×4`, which lands on a square `32×32` and required a fudge. Do not revert to that.)

### Critic (exact)

```python
class Critic(nn.Module):
    def __init__(self, cond_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(2 + cond_dim, 32, 4, 2, 1), nn.LeakyReLU(0.2),                       # [B,32,32,16]
            nn.Conv2d(32,  64, 4, 2, 1), nn.GroupNorm(1,  64), nn.LeakyReLU(0.2),          # [B,64,16,8]
            nn.Conv2d(64, 128, 4, 2, 1), nn.GroupNorm(1, 128), nn.LeakyReLU(0.2),          # [B,128,8,4]
            nn.Conv2d(128,256, 4, 2, 1), nn.GroupNorm(1, 256), nn.LeakyReLU(0.2),          # [B,256,4,2]
            nn.Flatten(), nn.Linear(256*4*2, 1),                                            # scalar, no sigmoid
        )
    def forward(self, x, c):
        # broadcast condition as constant spatial channels
        cmap = c.view(-1, c.shape[1], 1, 1).expand(-1, -1, x.shape[2], x.shape[3])
        return self.net(torch.cat([x, cmap], dim=1))
```

**Normalization note:** WGAN-GP requires **no BatchNorm in the critic** — BatchNorm makes the gradient penalty ill-defined because it couples samples within a batch. The reference implementation uses LayerNorm; `nn.GroupNorm(1, C)` is equivalent here and avoids `nn.LayerNorm`'s requirement to hardcode exact `[C,H,W]` shapes. Either is acceptable — **BatchNorm is not**.

### Gradient penalty (exact)

```python
def gradient_penalty(critic, real, fake, cond, device):
    B = real.size(0)
    eps = torch.rand(B, 1, 1, 1, device=device)
    interp = (eps * real + (1 - eps) * fake).requires_grad_(True)
    score = critic(interp, cond)
    grads = torch.autograd.grad(
        outputs=score, inputs=interp,
        grad_outputs=torch.ones_like(score),
        create_graph=True, retain_graph=True,
    )[0]
    return ((grads.view(B, -1).norm(2, dim=1) - 1) ** 2).mean()
```

### Mandatory shape verification before Phase 5

```python
g, d = Generator(), Critic()
z, c = torch.randn(4, 100), torch.rand(4, 2)*2 - 1
fake = g(z, c)
assert fake.shape == (4, 2, 64, 32), fake.shape
assert d(fake, c).shape == (4, 1)
print("shapes OK", sum(p.numel() for p in g.parameters()), sum(p.numel() for p in d.parameters()))
```

### Checkpoint
- [ ] Both assertions pass.
- [ ] `gradient_penalty` runs on a dummy batch and returns a finite scalar.
- [ ] A single manual train step (1 critic update + 1 generator update) executes without NaN.

### HANDOFF_PHASE_4.md must contain
1. Final verbatim code for `Generator`, `Critic`, `gradient_penalty`.
2. Printed parameter counts for both networks.
3. Confirmation output of the shape assertions.
4. Any deviation from the architectures above, with justification.

---

# PHASE 5 — WGAN-GP Training

### What to paste alongside this phase
- PROJECT CONTEXT BLOCK.
- `HANDOFF_PHASE_1.md` (data paths, position bounds) and **`HANDOFF_PHASE_4.md` in full** (the model code).

### Frozen hyperparameters (these are the published WGAN-GP defaults — do not substitute DCGAN values)
| Setting | Value | Why |
|---|---|---|
| `lambda_gp` | 10 | Gulrajani et al. standard |
| `n_critic` | 5 | 5 critic steps per generator step |
| Optimizer (both G and D) | `Adam(lr=1e-4, betas=(0.5, 0.9))` | **`beta2=0.9`, not 0.999** — DCGAN's 0.999 is specifically wrong for WGAN-GP |
| Batch size | 64 | |
| Iterations | 5000 generator steps (start here) | |
| `z_dim` | 100 | |

### Training loop skeleton

```python
for it in range(start_it, n_iters):
    for _ in range(n_critic):
        real, cond = next(loader)                      # real: [B,2,64,32], cond: [B,2]
        z = torch.randn(B, 100, device=dev)
        fake = G(z, cond).detach()
        gp = gradient_penalty(D, real, fake, cond, dev)
        d_loss = D(fake, cond).mean() - D(real, cond).mean() + 10.0 * gp
        optD.zero_grad(); d_loss.backward(); optD.step()

    z = torch.randn(B, 100, device=dev)
    g_loss = -D(G(z, cond), cond).mean()
    optG.zero_grad(); g_loss.backward(); optG.step()

    # Wasserstein estimate for the training-curve figure:
    #   w_est = D(real).mean() - D(fake).mean()
```

### Conditioning
Sample conditions by **drawing positions from the real seed set** (with replacement). This keeps the conditioning distribution identical to the real data and is the simpler, defensible choice. Document it.

### Checkpointing (required — this phase may cross a session boundary)
Every 500 iterations, save to `/kaggle/working/checkpoints/`:
```python
torch.save({'it': it, 'G': G.state_dict(), 'D': D.state_dict(),
            'optG': optG.state_dict(), 'optD': optD.state_dict(),
            'w_hist': w_hist}, f'/kaggle/working/checkpoints/gan_{it}.pt')
```
And **resume** from the latest checkpoint on script start.

### Monitoring — what healthy training looks like
- The **Wasserstein estimate** (`D(real) − D(fake)`) should generally *decrease* and stabilize. Oscillation is normal; a clean monotonic curve is not expected.
- `d_loss` and `g_loss` individually are **not** meaningful progress indicators in WGAN — do not panic at their raw values.
- **NaN at any point** = bug, not bad luck. Check: gradient penalty `create_graph=True`, no BatchNorm in critic, LR not raised.
- Save a grid of generated CSI magnitude heatmaps every 1000 iterations for visual tracking.

### If it collapses (all outputs near-identical)
In order: (1) verify GP code exactly matches Phase 4, (2) lower LR to `5e-5`, (3) raise `n_critic` to 10, (4) train longer. Do **not** switch architectures — a mediocre-but-honest GAN still yields a publishable ablation; the paper simply reports the quality metrics as they are.

### Checkpoint
- [ ] 5000 iterations completed (or documented early stop) without NaN.
- [ ] Wasserstein-estimate curve saved as CSV + plot.
- [ ] Final generator checkpoint saved and reloadable.
- [ ] Sample heatmap grids saved at several iteration counts.

### HANDOFF_PHASE_5.md must contain
1. Final training script verbatim.
2. Path to the final generator checkpoint (Phase 6 and 7 both load it).
3. The Wasserstein-estimate curve (CSV path + plot path) and a description of its shape.
4. Total training wall-clock time and iterations completed.
5. The exact conditioning-sampling procedure used.
6. An honest one-paragraph assessment: does the output look structured, or like noise/mode collapse?

---

# PHASE 6 — GAN Quality Check

### What to paste alongside this phase
- PROJECT CONTEXT BLOCK.
- `HANDOFF_PHASE_1.md`, `HANDOFF_PHASE_4.md`, `HANDOFF_PHASE_5.md`.

### Goal
Evidence — for the paper's Section IV-A — that the synthetic CSI is or is not realistic. **A negative result here is publishable; a hidden negative result is misconduct.** Report what you find.

### Test 1 — Qualitative heatmaps
Plot `|H| = sqrt(real² + imag²)` as a `64×32` heatmap for 8 real and 8 synthetic samples side by side, shared color scale. Save to `results/plots/real_vs_synth_heatmaps.png`. → **Figure 3** in the paper.

### Test 2 — Frobenius norm distribution
Overlaid histograms of `||H||_F` for real vs. synthetic (post-normalization these are ~1 by construction, so instead compare a **structural statistic**): use per-sample **power distribution across antennas** (`mean over subcarriers of |H[a,:]|²`, giving a 64-vector) and compare real vs. synthetic mean ± std curves. Also report **KL divergence or Wasserstein distance** between the two distributions of a scalar summary (e.g. antenna-domain power spread). → **Figure 4**.

### Test 3 — GAN-test discriminability (the key quantitative number)
Train a small binary classifier (reuse `BeamCNN` with `n_beams=2`) to distinguish real from synthetic:
- Balanced set: 500 real (seed set) + 500 synthetic.
- 80/20 train/test split, 3 seeds, report mean ± std accuracy.

**Interpretation, to be stated plainly in the paper:**
| Discriminator accuracy | Meaning |
|---|---|
| ~50–60% | Synthetic samples are near-indistinguishable — strong realism |
| 60–80% | Partially realistic; usable but note as a limitation |
| >90% | Easily distinguishable — report honestly; the ablation is still valid, and this becomes a central discussion point explaining any weak augmentation gains |

### Test 4 — Beam-label consistency (cheap and very informative)
Run the Phase 2 codebook labelling on synthetic samples. Compare the **synthetic label distribution** to the real one (histogram overlap). If the GAN produces CSI whose optimal beams are distributed like the real data, that is direct evidence it captured task-relevant structure. Report the overlap. → useful for the paper's discussion.

### Checkpoint
- [ ] All four tests run; figures saved.
- [ ] GAN-test discriminator accuracy recorded as mean ± std.
- [ ] An explicit written verdict: proceed / proceed-with-caveats / GAN is broken.

### HANDOFF_PHASE_6.md must contain
1. GAN-test discriminator accuracy (mean ± std) — a headline number for the paper.
2. Paths to all figures.
3. Real vs. synthetic beam-label distribution comparison.
4. Distributional distance metrics computed.
5. The written verdict and the reasoning behind it.

---

# PHASE 7 — Ablation Sweep (the paper's main experiment)

### What to paste alongside this phase
- PROJECT CONTEXT BLOCK.
- **`HANDOFF_PHASE_3.md` in full** (the frozen classifier + training function — reuse unmodified).
- `HANDOFF_PHASE_5.md` (generator checkpoint path) and `HANDOFF_PHASE_6.md` (quality verdict).

### The sweep — 7 ratios × 3 seeds = 21 runs (R0's 3 runs are already done in Phase 3, so 18 remain)

| Run | Real | Synthetic added | Total | Ratio |
|---|---|---|---|---|
| R0 | 500 | 0 | 500 | 0% |
| R25 | 500 | 125 | 625 | 25% |
| R50 | 500 | 250 | 750 | 50% |
| R75 | 500 | 375 | 875 | 75% |
| R100 | 500 | 500 | 1000 | 100% |
| R150 | 500 | 750 | 1250 | 150% |
| R200 | 500 | 1000 | 1500 | 200% |

**Convention (state this explicitly in the paper):** ratio = synthetic count as a percentage **of the real count**, *added on top of* the fixed 500 real samples. The real portion is identical in every run.

### Procedure per run
1. Load the frozen generator (eval mode, `torch.no_grad()`).
2. Sample `n_synth` conditions from the real seed positions (same procedure as Phase 5).
3. Generate `n_synth` synthetic CSI samples.
4. **Label them with the Phase 2 codebook function** — the GAN does not produce labels; the physics-based labeller assigns them. (This is a genuine methodological choice worth one sentence in the paper: labels for synthetic data are derived analytically from the generated channel, not predicted.)
5. Concatenate with the 500 real samples → training set.
6. Train `BeamCNN` with the **frozen Phase 3 recipe** — no hyperparameter changes.
7. Validation set: **real samples only**, unchanged across runs.
8. Evaluate once on the 2000-sample real test set; append the CSV row.

### Non-negotiables
- **Do not** re-tune LR/epochs per run.
- **Do not** put synthetic data in the validation or test sets.
- **Do** regenerate synthetic data fresh per seed (so the seed varies both weight init and the synthetic draw) — this is what makes the error bars meaningful.
- Run all 21 rows through the *same* code path, including R0, so nothing differs but the data.

### Budget
Per Phase 3's timing, each run is minutes. All 18 remaining runs should fit comfortably in one Kaggle session; checkpoint the CSV after every run regardless.

### Checkpoint
- [ ] `ablation_results.csv` contains 21 rows (7 ratios × 3 seeds).
- [ ] Aggregated mean ± std computed per ratio.
- [ ] Every run beats the majority-class baseline (if not, flag it prominently).

### HANDOFF_PHASE_7.md must contain
1. The complete `ablation_results.csv` contents (all 21 rows, pasted).
2. Aggregated table: ratio → Top-1 mean±std, Top-3 mean±std, SE% mean±std.
3. The synthetic-data generation + labelling code verbatim.
4. Total wall-clock time.
5. Any anomalies (diverged runs, unexpected orderings).

---

# PHASE 8 — Analysis, Figures & Tables

### What to paste alongside this phase
- PROJECT CONTEXT BLOCK.
- `HANDOFF_PHASE_2.md` (baseline), `HANDOFF_PHASE_6.md` (GAN quality), **`HANDOFF_PHASE_7.md` in full**.

### Deliverables — these map 1:1 onto the paper's placeholders

| Artifact | Paper slot | Spec |
|---|---|---|
| `fig_ablation_curve.png` | **Figure 5** (the key figure) | X = synthetic ratio (0–200%), Y = Top-1 accuracy, error bars = ±1 std over 3 seeds; dashed horizontal line = majority-class baseline; second series = Top-3 on a twin axis or a second panel |
| `fig_gan_training_curve.png` | Figure 2 | Wasserstein estimate vs. iteration |
| `fig_real_vs_synth.png` | Figure 3 | from Phase 6 |
| `fig_label_dist.png` | Figure 4 | real vs. synthetic beam-label histograms |
| `table_main_results.tex/.md` | **Table II** | ratio × {Top-1, Top-3, SE%} as mean±std |
| `table_hyperparams.md` | Table I | all frozen hyperparameters |

### Statistical analysis (do this — it is what makes the claim defensible)
1. For the best-performing ratio vs. R0, run a **two-sample t-test** (or Mann–Whitney U given n=3) on the per-seed Top-1 accuracies. Report the p-value.
2. With only 3 seeds, be candid: state that the seed count is small and that differences within one standard deviation are **not** claimed as significant.
3. Compute the **absolute improvement** (percentage points) from R0 to the best ratio — this is the abstract's headline number.

### Interpretation framework — decide which story the data actually tells
- **Monotonic increase:** augmentation helps throughout; note that the ceiling was not reached within 200%.
- **Rise then plateau:** identify the knee; that ratio is the practical recommendation.
- **Rise then decline:** synthetic data begins to dominate and injects GAN artifacts; the peak is the optimum and the decline is the interesting finding.
- **Flat / within noise:** a legitimate negative result. Connect it to the Phase 6 discriminability number — if the GAN was easily distinguishable, weak gains are the *expected* and explainable outcome. Frame this as an honest finding about the limits of GAN augmentation under extreme data scarcity, not as a failure.

### HANDOFF_PHASE_8.md must contain
1. Paths to every figure and table file.
2. The final aggregated results table, formatted.
3. The statistical test results (p-values).
4. A 200-word written interpretation identifying which of the four patterns above occurred.
5. The headline number for the abstract (e.g. "+X.X pp Top-1 at Y% ratio").

---

# PHASE 9 — Paper Assembly

### What to paste alongside this phase
- PROJECT CONTEXT BLOCK.
- **`PAPER_DRAFT.md`** (the skeleton — already written, with placeholders marked).
- **Every handoff file from Phases 1–8.**

### Task
Fill every `[[PLACEHOLDER]]` in `PAPER_DRAFT.md` with real values from the handoffs. Do **not** invent numbers. If a value is missing, mark it `[[MISSING — needs Phase N]]` and say so rather than guessing.

### Order of work
1. Fill Methodology (Section III) from Phases 1–5 handoffs — this is pure transcription.
2. Fill Results (Section IV) from Phases 6–8.
3. Write Discussion using the Phase 8 interpretation.
4. Write the Abstract **last**, once the headline number exists.
5. Verify every citation resolves (all references in the draft were verified against arXiv at drafting time).
6. Convert to the target venue's IEEE two-column template.

### Final integrity checklist
- [ ] Every number in the paper traces to a handoff file.
- [ ] Limitations section is honest (single scenario, single band, 500-sample seed, 3 seeds, GAN discriminability).
- [ ] No claim of significance that the statistics do not support.
- [ ] Author names/affiliations set per the venue template.
- [ ] All figures legible at print size in grayscale.

---

## Master progress tracker

| Phase | Owner | Status | Handoff written |
|---|---|---|---|
| 0 — Environment & scenario | | ⬜ | ⬜ |
| 1 — Data preprocessing | | ⬜ | ⬜ |
| 2 — Codebook & labels | | ⬜ | ⬜ |
| 3 — CNN + R0 baseline | | ⬜ | ⬜ |
| 4 — GAN architectures | | ⬜ | ⬜ |
| 5 — GAN training | | ⬜ | ⬜ |
| 6 — GAN quality check | | ⬜ | ⬜ |
| 7 — Ablation sweep | | ⬜ | ⬜ |
| 8 — Analysis & figures | | ⬜ | ⬜ |
| 9 — Paper assembly | | ⬜ | ⬜ |
