# GAN-Augmented mmWave Beamforming — Execution & Paper-Writing Guide

> **DOCUMENT SET** (four files, each with a distinct job):
> - `beamforming_project_spec.md` — original research spec: problem, motivation, novelty framing.
> - **`execution_and_paper_guide.md`** (this file) — the plan and the *reasoning behind it*, including §0.5 Implementation Notes & Errata (real-world API/environment gotchas — keep appending as you hit new ones).
> - `IMPLEMENTATION_PHASES.md` — **the build instructions.** Ten self-contained phases, each executable by a different AI with no prior context. Specifies what to paste into each phase and what handoff file each must produce. *Start here when writing code.*
> - `PAPER_DRAFT.md` — the paper, fully written except for `[[PLACEHOLDERS]]` covering only numbers, figures, and tables. References verified against arXiv.

> **PRECEDENCE — read this if two documents disagree.** These files were written at different times and overlap deliberately (redundancy is useful when phases are handed to different people/AIs). When they conflict:
> 1. **`IMPLEMENTATION_PHASES.md` wins on anything executable** — exact architectures, hyperparameters, code, file paths, run procedure. It is the newest and most specific.
> 2. **§0.5 of this file wins on environment/API facts** — it is the living log of what actually happened on real hardware.
> 3. This file's phase sections remain useful for **why** a choice was made, and for surrounding strategy (team split, timeline, venue selection) that `IMPLEMENTATION_PHASES.md` does not cover.
> 4. `beamforming_project_spec.md` is **historical context only.** Parts are known-superseded: it assumes an older DeepMIMO API, a 5-point ratio sweep (now 7), a pretrained-fine-tuned baseline "Run F" (dropped), a U-Net generator variant (dropped), and a generator whose shape math lands on `32×32` rather than `64×32` (corrected — do not use it).
>
> Rules added after the original plan and binding on all phases: **dead-user filtering** in Phase 1, a **majority-class baseline** from Phase 2 that every run must beat, and **validation sets containing real samples only** in every ablation run.

> **How to use this document:** This is the strategy-and-rationale document for the project, written as if one person is executing it start to finish. Your team of 3 should split the phases (suggested split at the end of each relevant phase), but everyone should read the whole thing once so nobody is blind to what the others are doing. Check off each "Checkpoint" before moving to the next phase — don't parallelize past a checkpoint you haven't hit.
>
> **Scope decisions locked in for this version** (see prior discussion):
> - ❌ Pretrained-fine-tuned baseline (old "Run F") — dropped. Not required.
> - ❌ U-Net skip-connection generator — dropped as a requirement. Plain conv-transpose generator only. (If someone finishes everything with days to spare, it's a nice-to-have stretch goal, never a blocker.)
> - ✅ Expanded ratio sweep — 7 points instead of 5, since more ratios = more training runs, not more architecture. This is now the paper's main empirical contribution.
> - Target: real, indexed, fast-turnaround venue (regional/student IEEE conference or similar) — not a top-tier journal, not a predatory pay-to-publish outlet.

---

## 0.5 Implementation Notes & Errata (living log — read this before touching DeepMIMO)

> This section exists because the original plan below (written before any code was run) assumed an older DeepMIMO API. Reality diverged in specific, documented ways. **If you're picking this project up fresh (including a different AI assistant), read this section first — it overrides anything in the phases below that conflicts with it.** Keep appending to this section as new gotchas are discovered; don't let tribal knowledge live only in a chat log.

**DeepMIMO version actually installed:** `4.0.3` (via `pip install DeepMIMO`). This is a different, restructured API generation from what older tutorials/the original spec assumed (which referenced `DeepMIMOv3`-style dict configs). Concretely:

- **Import name is lowercase:** `import deepmimo` (not `import DeepMIMO`), despite the pip package being capitalized `DeepMIMO`.
- **Restart the kernel after `pip install DeepMIMO`** before importing — it will `ModuleNotFoundError` otherwise even though install succeeded. Standard Jupyter/Kaggle behavior, not a DeepMIMO bug.
- **Scenario discovery:** `deepmimo.get_available_scenarios()` only lists *locally downloaded* scenarios (starts empty). To find scenario names in the online catalog, use `deepmimo.search(query_dict)`, e.g. `deepmimo.search({'bands': ['mmW']})`. This is also how you find exact scenario name variants — "O1" is not a valid name by itself; the actual mmWave/28GHz O1 scenario is named **`o1_28`** (lowercase, with band suffix). Other relevant names found this way: `o1_28`, `o1b_28`, `o1_60`, `boston5g_28`.
- **Relevant top-level API surface** (from `dir(deepmimo)`): `download()`, `search()`, `load()`, `generate()`, `ChannelParameters`, `Dataset`, `steering_vec()` (this last one may let us skip manually deriving the DFT beam codebook in Phase 2 — check `help(deepmimo.steering_vec)` when we get to that phase, it might already do what `utils/codebook.py` was going to hand-roll).
- **`deepmimo.download(scenario_name, output_dir=None, rt_source=False)`** — downloads the compiled scenario (not raw ray-tracing source, which is what we want; leave `rt_source=False`).

**Kaggle disk layout — this matters a lot, don't skip it:**
- `/kaggle/working` is a **20GB hard-capped loop-device mount** — this is Kaggle's *persistent/committed output* quota, not general scratch space. The full `o1_28` scenario does **not** fit when extracted here (a 2.28GB zip expands to something larger than 20GB once unzipped — extraction failed with `OSError: No space left on device` at 57% through).
- The **root filesystem (`overlay`, mounted at `/`) has ~1.1TB free** on a typical Kaggle GPU session. Download/extract the raw DeepMIMO scenario there instead, e.g. `/root/deepmimo_scenarios`, via `deepmimo.download('o1_28', output_dir='/root/deepmimo_scenarios')`.
- **Consequence for the pipeline:** raw scenario data lives outside `/kaggle/working` and does **not** persist across sessions (neither does `/kaggle/working`'s non-committed contents, but `/root/...` doesn't even have the 20GB commit option). Workflow per session: download+extract raw scenario to `/root/...` → subsample/preprocess down to the small `[N,2,64,32]` tensors we actually need (Phase 1 steps 3-9) → save *only those small tensors* (a few MB, not GB) into `/kaggle/working` or a Kaggle Dataset → the multi-GB raw scenario can be deleted or just left to be wiped at session end, no need to keep it.
- **If a download succeeds but extraction fails, don't delete the zip** — only delete the failed/partial extraction output and re-extract from the existing zip. Re-downloading a multi-GB file you already have wastes time and can trigger rate limiting (see next point).
- **`output_dir` does NOT control where the scenario ends up (confirmed on 4.0.5, 2026-10-14).** This is the most expensive gotcha found so far. `deepmimo.download('o1_28', output_dir='/root/deepmimo_scenarios')` downloads the zip there and extracts there (both succeed — 2.28 GB, 4161 files), then `_extract_and_move_scenario()` calls `shutil.move()` to the library's **configured scenarios directory**, which defaults to `/kaggle/working/deepmimo_scenarios/`. Because `/root` and `/kaggle/working` are different devices, `os.rename` raises `OSError: [Errno 18] Invalid cross-device link`, `shutil` silently falls back to `copytree`, and the copy then dies with `[Errno 28] No space left on device` against the 20 GB cap. **The download is not the problem; the install target is.**
  - **Recovery, in this order:** (1) `rm -rf /kaggle/working/deepmimo_scenarios` — the partial copy has filled your 20 GB quota and nothing works until it is gone. (2) Verify `/root/deepmimo_scenarios/o1_28` still has 4161 files — `copytree` raises before `rmtree(src)` runs, so the source survives. (3) **Symlink** rather than copy: `os.symlink('/root/deepmimo_scenarios/o1_28', '/kaggle/working/deepmimo_scenarios/o1_28')`, then confirm with `deepmimo.get_available_scenarios()`. Alternatively repoint the library's scenarios directory (inspect `deepmimo.config`) at `/root/deepmimo_scenarios` and keep raw data off `/kaggle/working` entirely.
  - **Measured sizes (2026-10-14):** the 2.28 GB zip extracts to **21 GB** across 4161 files. That is larger than the entire 20 GB `/kaggle/working` cap, so the scenario can **never** live there even on a completely empty quota. The symlink is not a workaround for a tight fit — it is the only viable approach. Confirmed working: after symlinking, `deepmimo.get_available_scenarios()` returns `['o1_28']`.
  - **Do not re-run `download()`** after this failure — you already have both the zip and the extract, and re-downloading risks the 429 below for no gain.
- **Installed version drifts.** 4.0.5 was observed on 2026-10-14 where this document previously recorded 4.0.3. Print `deepmimo.__version__` and don't assume signatures match an older note.
- **DeepMIMO's download server rate-limits (`429 Too Many Requests`)** if you retry downloads in quick succession. If you hit this, stop retrying immediately — wait 10-15 minutes before trying again once. Don't loop retries in code.

**API facts established 2026-10-14 (DeepMIMO 4.0.5) — these resolve most of the old open questions:**

- **`load()` returns a `MacroDataset`, not a `Dataset`.** `deepmimo.load('o1_28')` with no restriction loads **all 54 TX-RX pairs** (18 TX sets, TXset 3–20, × 3 RX sets) and takes minutes. RX set sizes are **497,931 / 199,281 / 487,711** users, i.e. ~1.18M users per TX set. Restrict with `load_params={'tx_sets': ..., 'rx_sets': ...}` — the project needs 2,500 users, so one TX set and one RX set is ample. Index a loaded MacroDataset via `ds.datasets[i]`; `len(ds)` is the pair count.
- **`generate()` is the channel-computing entry point and supports UE subsetting.** Signature: `generate(scen_name, *, load_params=None, trim_params=None, ch_params=None)`. `trim_params` accepts `idxs` (UE indices, **applied first**), `idxs_mode` ∈ `'active'|'linear'|'uniform'|'row'|'col'|'limits'`, `idxs_kwargs`, plus `bs_fov`, `ue_fov`, `path_depth`, `path_types`. **Use this instead of loading everything and slicing afterwards** — it is the difference between a fast Phase 1 and an out-of-memory fight. `idxs_mode='active'` is also the natural way to get the dead-user filtering that Phase 1 requires. Helpers: `get_uniform_idxs`, `get_linear_idxs`, `get_idxs_with_limits`, `get_grid_idxs`.
- **`load_params` also takes `max_paths`** (default 10) and `matrices` (restrict which matrices load).
- **`ChannelParameters` is a `DotDict`** — set nested values as dicts, e.g. `ChannelParameters(bs_antenna={'shape': [64, 1]}, freq_domain=True)`, and read them back with dot notation (`params.bs_antenna.shape`). Lists become numpy arrays at validation. `params.validate(n_ues)` checks consistency. Inspect the full default set with `pprint(ChannelParameters().to_dict())`.
- **`steering_vec()` does NOT replace the hand-rolled DFT codebook — decided, keep the hand-rolled version.** Its signature is `steering_vec(array, phi=0, theta=0, spacing=0.5)`: it takes antenna *positions* and a single direction and returns one array-response vector, so building a 32-beam codebook would mean 32 calls with hand-mapped angles — more bug surface than the five-line DFT formula, not less. The DFT codebook is also the standard definition in the beam-prediction literature and therefore the better thing to report.

**Still to be confirmed once we get further into Phase 1** (update this list as we go):
- The exact **OFDM field names inside `ChannelParameters.DEFAULT_PARAMS`**, and whether `o1_28` supports exactly **32 subcarriers**. If it does not, the nearest supported value must be documented and propagated to every `32` downstream (codebook size is separate and stays 32).
- The attribute names on a single `Dataset` (`ds.datasets[i]`) holding the **channel matrix** and **RX positions**.
- Scenario carrier frequency and position grid bounds, for the paper's methodology section.

*Resolved 2026-10-14 (see the API facts above): `load()` vs `generate()` signatures, `MacroDataset` structure, UE subsetting via `trim_params`, and the `steering_vec()` question (answer: keep the hand-rolled DFT codebook).*

---

## 0. Kaggle / GPU Session Strategy (read this before writing any code)

**The constraint is session length (7 hrs), not total hours (you have ~120/week).** Design around that:

1. **Every training script must support checkpoint + resume.** Save `state_dict()` for generator, critic, and optimizer every N epochs (or every X minutes) to `/kaggle/working/checkpoints/`. At the top of every training script, check if a checkpoint exists and resume from it instead of starting over.
2. **Download checkpoints/outputs from `/kaggle/working/` before the session ends.** Kaggle wipes the working directory between sessions unless you commit the notebook version or manually save to a Kaggle Dataset. Do this after every session — losing a checkpoint because you forgot to save the notebook output is the single most common way people waste Kaggle hours.
3. **One phase per session, roughly:**
   - Session 1: Data pipeline + codebook (Phase 1-2) — should take under an hour, leaves room to also run the baseline CNN (Phase 3).
   - Session 2: GAN implementation + shape debugging (Phase 4) — mostly CPU-side debugging, GPU only needed for the actual training runs.
   - Session 3: GAN training to convergence (Phase 5) — this is the only phase likely to approach a full session; checkpoint every 500 iterations.
   - Session 4: Ablation sweep, all 7 ratio runs (Phase 7) — each CNN run is minutes, so this whole phase should fit in one session with room to spare.
   - Session 5: Buffer / re-runs / quality-check plots.
4. Realistically you will not come close to using your 120-hour weekly budget. Don't over-engineer around it — just checkpoint sanely and you're fine.

---

## 1. Project Summary (condensed)

Train a conditional WGAN-GP on a small seed set (500 real samples) of mmWave CSI from DeepMIMO, conditioned on user position. Use it to generate synthetic CSI at increasing ratios relative to the real set. Train a CNN beam classifier at each ratio and measure how synthetic augmentation affects beam-prediction accuracy. The ratio-sweep curve *is* the paper's result.

---

## 2. Environment Setup (Phase 0)

**Checklist:**
- [ ] Kaggle notebook created, GPU (T4) accelerator enabled.
- [ ] `pip install DeepMIMO` (verify this is still the correct package name/API at implementation time — check https://github.com/DeepMIMO/DeepMIMO-python; APIs on data-generation packages do drift).
- [ ] `pip install torch numpy matplotlib seaborn scikit-learn pandas tqdm` (torch is pre-installed on Kaggle GPU images — just verify version).
- [ ] Create a Kaggle Dataset (not just notebook output) to persist: **preprocessed tensors, model checkpoints, and results CSVs only**. Do **not** try to persist the raw DeepMIMO extract — it exceeds the 20 GB `/kaggle/working` cap (see §0.5); re-download it to `/root/...` each session instead, or better, only once during Phase 1 since nothing after Phase 2 needs it.
- [ ] Set and log a fixed random seed (`torch.manual_seed(42)`, `np.random.seed(42)`) for every script — you'll want reproducibility when writing the paper.

**Repo/notebook structure** (same as before, keep it):
```
mmwave-gan-beamforming/
├── data/            # download_deepmimo.py, preprocess.py
├── models/          # generator.py, critic.py, beam_classifier.py
├── training/        # train_gan.py, train_classifier.py
├── evaluation/       # metrics.py, gan_quality_check.py, run_ablation.py
├── utils/           # codebook.py, data_utils.py
├── notebooks/        # kaggle_main.ipynb
└── results/          # ablation_results.csv, plots/
```

---

## 3. Phase 1 — Data Pipeline

**Goal:** end up with tensors of shape `[N, 2, 64, 32]` (CSI) + `[N, 2]` (position) + `[N]` (beam label, computed in Phase 2).

**Steps:**
1. Load the DeepMIMO **`o1_28`** scenario (lowercase, band-suffixed — see §0.5; `O1` alone is not a valid name). Chosen because the O1 family is the most commonly cited in beam-prediction literature.
2. Configure: 28 GHz band, 64-element ULA at the base station, 32 OFDM subcarriers. (If DeepMIMO's generator for O1 doesn't support exactly 32 subcarriers out of the box, use the nearest supported value and document the actual number used — don't fight the tool, just report what you used.)
3. Extract complex channel matrix `H` per user: shape `[64, 32]` complex.
4. Extract `(x, y)` position per user (drop `z` — O1 is a ground-level scenario, elevation isn't meaningfully varying).
5. Convert `H` → real-valued tensor: `torch.stack([H.real, H.imag], dim=0)` → shape `[2, 64, 32]`.
6. **Normalize** each sample by its Frobenius norm: `H_norm = H / ||H||_F`. Store the original norm separately if you need to denormalize later for spectral efficiency (Phase 8) — decide now and write it down, don't rediscover this need mid-Phase-8.
7. Normalize positions to `[-1, 1]` using the scenario's known coordinate bounds (DeepMIMO scenario configs specify grid bounds).
8. **Split:**
   - Real seed set: random sample of **500** users → this is your scarce "real" training pool for all downstream runs.
   - Held-out test set: **2000** users from positions NOT in the seed set → touched only at final evaluation, never during any training (GAN or CNN).
9. Save all tensors as `.pt` files (`torch.save`) to your Kaggle Dataset.

**Checkpoint:** You can load a batch and get `csi.shape == [B, 2, 64, 32]`, `pos.shape == [B, 2]`.

---

## 4. Phase 2 — Beam Codebook & Ground-Truth Labels

**DFT Codebook** (`utils/codebook.py`):
- `N_beams = 32`.
- For a 64-element ULA, the k-th DFT beamforming vector is:
  `w_k[n] = (1/sqrt(64)) * exp(j * 2 * pi * n * k / N_beams)`, for `n = 0..63`, `k = 0..31`.
- Build this as a `[32, 64]` complex matrix (`codebook`), computed once, reused everywhere.

**Optimal beam label per sample:**
- For channel `H` (shape `[64, 32]`, complex), compute received power per beam per subcarrier: `power[k, s] = |conj(w_k) @ H[:, s]|^2`.
- Average (or sum) over subcarriers to get a scalar score per beam: `score[k] = mean_s(power[k, s])`.
- Ground-truth label = `argmax_k score[k]`.
- Store this as an integer label `[N]` alongside the CSI tensors.

**Checkpoint:** Every sample in both the seed set and test set has a beam label in `[0, 31]`, and labels are reasonably spread across the 32 classes (sanity-check with a histogram — if 90% of labels are the same 2 beams, something's wrong with your codebook math or the scenario geometry).

---

## 5. Phase 3 — CNN Beam Classifier + Run @ 0% (baseline)

**Exact architecture** (`models/beam_classifier.py`):

```
Input: [B, 2, 64, 32]

Conv2d(2, 32, kernel_size=3, padding=1) -> BatchNorm2d(32) -> ReLU
Conv2d(32, 64, kernel_size=3, padding=1) -> BatchNorm2d(64) -> ReLU -> MaxPool2d(2)
    # shape now: [B, 64, 32, 16]
Conv2d(64, 128, kernel_size=3, padding=1) -> BatchNorm2d(128) -> ReLU -> MaxPool2d(2)
    # shape now: [B, 128, 16, 8]
Flatten -> [B, 128*16*8] = [B, 16384]
Linear(16384, 256) -> ReLU -> Dropout(0.3)
Linear(256, 32)   # logits over N_beams=32, NO softmax here — CrossEntropyLoss applies it
```

- Params: ~4.3M (dominated by the first Linear layer — this is fine, still trains in seconds/epoch on a T4).
- Loss: `nn.CrossEntropyLoss()`.
- Optimizer: `Adam(lr=1e-3)`.
- Scheduler: **`CosineAnnealingLR`** over total epochs (frozen choice — `ReduceLROnPlateau` was the alternative, but the recipe must be identical across all 21 runs or the ablation is confounded).
- Epochs: **100** (frozen — see `IMPLEMENTATION_PHASES.md` Phase 3 for the authoritative recipe). With only 500 samples this will overfit fast; that's expected and fine. Use best-validation-accuracy checkpointing and evaluate on the held-out test set, not the training curve.
- Train/val split within the 500 real samples: e.g. 400 train / 100 val (val used for early stopping / LR scheduling only, never for the final reported numbers — those come from the 2000-sample held-out test set).

**This training run — real data only, 0% synthetic — is Run R0, your first ablation data point.**

**Checkpoint:** Full training loop runs, you have Top-1 and Top-3 accuracy on the held-out test set for Run R0. This is the most important early checkpoint in the whole project — if this pipeline (data → train → eval) works, everything after this is "run the same thing on different data mixes."

**Suggested owner:** Person A.

---

## 6. Phase 4 — WCGAN-GP: Generator & Critic

### 6.1 Generator (`models/generator.py`)

This is a corrected/exact version of the earlier architecture — computed so the output shape lands on `[2, 64, 32]` exactly, no cropping/padding hacks needed.

```
Input: noise z ~ N(0,1), dim=100
       condition c = (x, y) position, dim=2
       concat: [z; c] -> dim 102

Linear(102, 256*4*2 = 2048) -> reshape to [B, 256, 4, 2]

ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1) -> BatchNorm2d -> ReLU
    # [B, 128, 8, 4]
ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1) -> BatchNorm2d -> ReLU
    # [B, 64, 16, 8]
ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1) -> BatchNorm2d -> ReLU
    # [B, 32, 32, 16]
ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1) -> BatchNorm2d -> ReLU
    # [B, 16, 64, 32]
Conv2d(16, 2, kernel_size=3, stride=1, padding=1) -> Tanh
    # [B, 2, 64, 32]  <- exact target shape, verified by the stride-2 doubling math
```

**Why this version differs from the original draft spec:** the original started from a `4x4` seed and used 3 symmetric doubling layers, which lands on `32x32` — square, but your target is `64x32` (2:1 aspect ratio), which needed an awkward crop/pad fudge. Starting from a `4x2` seed and using **4** doubling layers (`4→8→16→32→64` on the antenna axis, `2→4→8→16→32` on the subcarrier axis) lands exactly on `[64, 32]` with clean stride-2/padding-1 arithmetic. Verify with a dummy forward pass before trusting this — always do this before committing to a real training run:

```python
g = Generator()
z = torch.randn(4, 100)
c = torch.randn(4, 2)
out = g(z, c)
assert out.shape == (4, 2, 64, 32)
```

### 6.2 Critic (`models/critic.py`)

Condition injection: broadcast `(x, y)` as two constant-valued spatial channels and concatenate with the 2 CSI channels → 4 input channels total.

```
Input: CSI [B, 2, 64, 32] + condition broadcast to [B, 2, 64, 32] -> concat -> [B, 4, 64, 32]

Conv2d(4, 32, kernel_size=4, stride=2, padding=1) -> LeakyReLU(0.2)
    # [B, 32, 32, 16]   (no norm on first layer — standard WGAN-GP practice)
Conv2d(32, 64, kernel_size=4, stride=2, padding=1) -> LayerNorm([64,16,8]) -> LeakyReLU(0.2)
    # [B, 64, 16, 8]
Conv2d(64, 128, kernel_size=4, stride=2, padding=1) -> LayerNorm([128,8,4]) -> LeakyReLU(0.2)
    # [B, 128, 8, 4]
Conv2d(128, 256, kernel_size=4, stride=2, padding=1) -> LayerNorm([256,4,2]) -> LeakyReLU(0.2)
    # [B, 256, 4, 2]
Flatten -> [B, 256*4*2 = 2048]
Linear(2048, 1)   # scalar critic score, no sigmoid
```

**Practical note on LayerNorm over conv feature maps:** `nn.LayerNorm` needs `normalized_shape` to match the exact `[C, H, W]` at that point (shown above) — it will not broadcast like BatchNorm does. If that's fiddly to get right, `nn.GroupNorm(1, C)` is the standard drop-in substitute used in most public WGAN-GP implementations and is normalization-equivalent to LayerNorm here — use that if you hit shape errors.

### 6.3 WGAN-GP Training Specifics (non-negotiable, these are the standard-correct values)

- Gradient penalty: `lambda_gp = 10`.
- `n_critic = 5` (5 critic updates per 1 generator update).
- Optimizer: `Adam(lr=1e-4, betas=(0.5, 0.9))` for **both** generator and critic. (Not `betas=(0.5, 0.999)` — that's the DCGAN default and is specifically wrong for WGAN-GP.)
- Gradient penalty implementation:
  ```python
  eps = torch.rand(B, 1, 1, 1)
  interp = eps * real + (1 - eps) * fake
  interp.requires_grad_(True)
  score = critic(interp, condition)
  grads = torch.autograd.grad(score.sum(), interp, create_graph=True)[0]
  gp = ((grads.view(B, -1).norm(2, dim=1) - 1) ** 2).mean()
  critic_loss = fake_score.mean() - real_score.mean() + lambda_gp * gp
  ```
- Batch size: 64 (fine for 500 samples — you'll just cycle through the seed set repeatedly with shuffling; that's normal and expected for GAN training on a small seed set).

**Checkpoint:** dummy forward pass shapes check out (Section 6.1), training loop runs a few hundred iterations without NaNs, Wasserstein estimate (`real_score.mean() - fake_score.mean()`) is trending in a sane direction.

**Suggested owner:** Person B.

---

## 7. Phase 5 — GAN Training to Convergence

- Train for a few thousand iterations (start with 3000-5000, watch the loss curves — GAN "convergence" here means stable, not a clean textbook curve; some oscillation is normal for WGAN-GP).
- Checkpoint generator + critic + optimizers every 500 iterations to your Kaggle Dataset (see Section 0 — this is the phase most likely to span a session boundary).
- Log the Wasserstein distance estimate per iteration for the paper's training-curve figure.

**Checkpoint:** Loss curves are logged, checkpoints saved, generator produces non-degenerate output (visually inspect a few samples before moving on — if everything looks like uniform noise or a single repeated pattern, that's mode collapse, see Section 15 mitigations from the original spec: lower LR, increase `n_critic`, double check the gradient penalty math above).

---

## 8. Phase 6 — GAN Quality Check

Before using synthetic data downstream, verify it's not garbage:

1. **Qualitative:** plot magnitude heatmaps (`|H|`) for a handful of real vs. synthetic samples side by side, same condition/position where possible.
2. **Quantitative — Frobenius norm distributions:** histogram of `||H||_F` for real vs. synthetic sets (before normalization, or compare the normalization-invariant structure — decide and document which).
3. **Quantitative — GAN-train/GAN-test check:** train a small binary classifier (reuse a shrunk version of the CNN, or even simpler — logistic regression on flattened features) to distinguish real vs. synthetic samples. If it can't beat ~55-60% accuracy, that's a good realism signal. If it's at 95%+, your synthetic samples are easily distinguishable — worth noting honestly in the paper as a limitation rather than hiding it.

**Checkpoint:** You have the plots + numbers for the paper's "GAN quality" subsection, and you've made an honest call on whether the GAN output is good enough to proceed (it doesn't need to be perfect — even a "70% GAN-test accuracy" result is a legitimate, reportable outcome, just be honest about it in the discussion).

**Suggested owner:** Person B (same person who owns the GAN, since they'll debug fastest).

---

## 9. Phase 7 — Ablation Sweep (the paper's main result)

**Expanded ratio sweep — 7 runs, all using the exact same 500-sample real seed set as the fixed "real" portion:**

| Run | Real samples | Synthetic samples added | Total | Synthetic ratio |
|---|---|---|---|---|
| R0   | 500 | 0   | 500  | 0%   |
| R25  | 500 | 125 | 625  | 25%  |
| R50  | 500 | 250 | 750  | 50%  |
| R75  | 500 | 375 | 875  | 75%  |
| R100 | 500 | 500 | 1000 | 100% |
| R150 | 500 | 750 | 1250 | 150% |
| R200 | 500 | 1000| 1500 | 200% |

(Convention: "ratio" = synthetic count as a percentage **of the real count**, added on top — state this explicitly in the paper, Section 10's original note about picking one convention and sticking to it still applies.)

**R0 is already done** (Phase 3). For R25 through R200:
1. Generate the required number of synthetic samples from the trained generator, conditioned on positions sampled from the same distribution as your real seed set's positions (or reuse real seed positions as conditions — pick one and document it; reusing real positions is simpler and defensible).
2. Combine real + synthetic into the training set for that run.
3. Train the **same** CNN classifier architecture (Phase 3) with the **same** hyperparameters — only the training data changes. This is critical for the ablation to be a clean comparison; don't tune hyperparameters per-run.
4. Evaluate on the **same** held-out 2000-sample real test set.
5. Log: Top-1 accuracy, Top-3 accuracy, spectral efficiency (Phase 8 below) to `results/ablation_results.csv`.

**Statistical rigor — do this, it's cheap and it's what separates a defensible result from a coin flip:** run each of the 7 conditions with **at least 3 different random seeds** (different weight init + different train/val shuffling), report mean ± std. This is a few dozen extra minutes of compute total (each CNN run is minutes), not a scheduling problem, and it's the single highest-value thing you can do to make the results table credible to a reviewer.

**Checkpoint:** `results/ablation_results.csv` has 7 rows × 3 seeds = 21 runs' worth of raw numbers, aggregated to mean±std per ratio.

**Suggested owner:** Person A (owns the CNN training loop already) with Person B feeding synthetic data batches.

---

## 10. Phase 8 — Results Analysis & Plotting

**Metrics** (`evaluation/metrics.py`):
1. **Top-1 accuracy:** `mean(pred_label == true_label)`.
2. **Top-3 accuracy:** `mean(true_label in top3(pred_logits))`.
3. **Spectral efficiency (% of optimal):**
   - For each test sample, compute achievable rate using the *predicted* beam: `SE_pred = log2(1 + SNR_pred)`, where `SNR_pred` derives from `|w_pred^H H|^2` (denormalize using the stored Frobenius norm from Phase 1 if you normalized).
   - Compute the optimal achievable rate `SE_opt` the same way using the ground-truth optimal beam.
   - Report `mean(SE_pred / SE_opt) * 100%` across the test set.

**Key plot:** X-axis = synthetic ratio (0, 25, 50, 75, 100, 150, 200%), Y-axis = Top-1 accuracy, with error bars from the 3-seed std (Section 9). This single figure is the paper's central result — spend real time making it clean (matplotlib/seaborn, labeled axes, legible at print size).

**Results table format:**

| Run | Ratio | Top-1 Acc (mean±std) | Top-3 Acc (mean±std) | Spectral Eff. % (mean±std) |
|---|---|---|---|---|
| R0 | 0% | | | |
| R25 | 25% | | | |
| ... | ... | | | |
| R200 | 200% | | | |

**Checkpoint:** Table + plot done, and you've looked at the trend and formed an actual interpretation (does accuracy monotonically improve, plateau, or peak-and-decline past some ratio? — any of these is a valid, interesting result, you just need to say which one you got and why you think it happened).

**Suggested owner:** Person C, once numbers land from A/B.

---

## 11. Phase 9 — How to Write the Paper

Target format: standard IEEE conference two-column template (most regional/student IEEE conferences use this — grab the template from the specific conference's site once you've picked one). Typical length: 4-6 pages.

**Section-by-section:**

### Title
Something close to: *"GAN-Augmented Channel Synthesis for Data-Efficient mmWave Beamforming: An Ablation Study on Synthetic-to-Real Data Ratios"* — descriptive, states the method and the contribution in one line. Fine as-is.

### Abstract (150-250 words)
One sentence each: problem (data scarcity in mmWave beam prediction) → approach (conditional WGAN-GP synthesizes CSI, ablate synthetic ratio) → what you measured (Top-1/Top-3 accuracy, spectral efficiency across 7 ratios) → headline result (state the actual number once you have it, e.g. "accuracy improves from X% to Y% at Z% synthetic ratio, plateauing beyond that") → one-line significance ("demonstrates synthetic augmentation is a viable mitigation for CSI data scarcity"). Write this **last**, after Phase 8 results are in — don't draft it from guesses now. *A full abstract template is already written in `PAPER_DRAFT.md`.*

### I. Introduction (~0.75-1 page)
- Para 1: why mmWave/6G beamforming matters, why it needs ML-based beam prediction (cite Alrabeiah & Alkhateeb).
- Para 2: the data scarcity problem — real CSI measurement is expensive, gets worse at higher frequencies for 6G.
- Para 3: synthetic data / GANs as a candidate mitigation, but the ratio question is understudied — this is your gap.
- Para 4: contributions, as a bulleted list: (1) conditional WGAN-GP for CSI synthesis conditioned on position, (2) systematic 7-point ablation of synthetic ratio, (3) [if you kept it] quality-check methodology for validating synthetic CSI realism.

### II. Related Work (~0.75-1 page)
**This section is already fully written in `PAPER_DRAFT.md` §II** — five subsections (A: beam prediction; B: wireless data scarcity/augmentation; C: generative models and alternatives; D: the ratio question in other domains; E: positioning), backed by **14 references, every one verified against arXiv**. You should not need to rewrite it, only to re-check that nothing important appeared after drafting.

**Critical framing point discovered during the literature search — do not lose it:** the synthetic-to-real ratio question *has* been studied in other domains, notably GAN-augmented medical imaging, where the optimum sat at a **small** synthetic fraction and performance *declined* as synthetic data grew. This means the novelty claim must be scoped as **"first ratio sweep for mmWave beam prediction"**, not "nobody has ever studied augmentation ratios." Claiming the latter is both false and easily caught by a reviewer. The cross-domain finding is an asset, not a threat: it gives the paper a concrete hypothesis to test and a comparison point for the discussion.

### III. Methodology
- III-A. Dataset & preprocessing (Phase 1) — scenario, band, antenna config, subcarriers, normalization, seed/test split sizes.
- III-B. Beam codebook & labels (Phase 2) — DFT codebook formula, optimal-beam label definition.
- III-C. WCGAN-GP architecture (Phase 4) — include the architecture diagrams/tables from Section 6 above, plus the loss function and hyperparameters.
- III-D. CNN beam classifier (Phase 3) — architecture table, hyperparameters.
- III-E. Ablation design (Phase 7) — the ratio table, the statistical-seed protocol.

### IV. Experiments & Results
- IV-A. GAN quality check results (Phase 6) — the heatmaps, Frobenius comparison, GAN-test accuracy number.
- IV-B. Ablation results — the table and the key plot from Phase 8.
- IV-C. Discussion — interpret the trend. Does more synthetic data help monotonically? Plateau? This is where you demonstrate understanding, not just report numbers.

### V. Conclusion
2-3 sentences: restate what you did, restate the headline finding, one sentence on limitations (single scenario, single frequency band — be upfront, reviewers respect this more than overclaiming) and one sentence on future work (e.g. "extending to multiple scenarios" or "the U-Net variant explored as future work" — you can honestly mention U-Net here as a natural extension without having built it).

### References
**Already done — 14 references in `PAPER_DRAFT.md` §IX, all verified against arXiv** (title, author list, and year confirmed by retrieving each abstract page). That count is appropriate for a 4–6 page paper. If you add more, verify them the same way: **never cite a paper you have not opened.** Fabricated or misattributed references are among the fastest routes to desk rejection, and LLM-generated bibliographies are a known source of them.

**Writing process tips:**
- Draft Methodology and Related Work **before** results land (Person C, in parallel per the Phase 0 team split) — these don't depend on your numbers.
- Draft Abstract and Conclusion **last**.
- Use a reference manager (Zotero/Mendeley, or even just a shared spreadsheet) from day one so citation formatting isn't a last-day scramble.
- Get all 3 team members' names + affiliations formatted per the target venue's template early — this is a common last-minute fire drill.

---

## 12. Phase 10 — Venue Selection & Submission

- Pick a specific regional/student IEEE conference (or equivalent) with a submission deadline that gives you buffer after Week 3-4. Check: indexed on IEEE Xplore, has a real (even if light) peer review, reasonable/transparent fee, not on a predatory-journal watchlist.
- Format to their exact template before submitting (margins/columns/reference style vary by publisher even within "IEEE format").
- Submit, and also post to arXiv in parallel if you want a citable preprint immediately rather than waiting on conference review timelines.

---

## 13. Master Checklist

- [ ] Phase 0: Kaggle environment + checkpoint/resume strategy working.
- [ ] Phase 1: Data pipeline, seed set (500) + test set (2000) separated.
- [ ] Phase 2: DFT codebook + beam labels computed, sanity-checked (label histogram not degenerate).
- [ ] Phase 3: CNN classifier trained on real-only data — Run R0 done, numbers in hand.
- [ ] Phase 4: Generator + critic implemented, dummy forward pass shapes verified.
- [ ] Phase 5: WGAN-GP trained, checkpointed, loss curves logged.
- [ ] Phase 6: GAN quality check done (plots + Frobenius comparison + GAN-test accuracy).
- [ ] Phase 7: All 7 ratio runs × 3 seeds completed, results logged.
- [ ] Phase 8: Metrics computed, results table + key plot generated.
- [ ] Phase 9: Paper drafted, all sections, formatted to target venue template.
- [ ] Phase 10: Venue picked, submitted.
