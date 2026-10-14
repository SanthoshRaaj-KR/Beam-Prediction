# HANDOFF_PHASE_1 — Data pipeline

Run on Kaggle, 2026-10-14, DeepMIMO 4.0.5, scenario `o1_28`.
Produced by `run_all.py` stage `data` in **14 s**.

## 1. ChannelParameters configuration (verbatim)

```python
p = deepmimo.ChannelParameters()
p.bs_antenna.shape = np.array([64, 1])    # 64-element ULA (library default is [8,1]!)
p.ue_antenna.shape = np.array([1, 1])     # single-antenna RX
p.ofdm.subcarriers = 512                  # FFT size
p.ofdm.selected_subcarriers = np.linspace(0, 511, 32).astype(int)
p.freq_domain = 1
p.num_paths = 25
```

Generated with:

```python
deepmimo.generate(
    "o1_28",
    load_params={"tx_sets": {4: "all"}, "rx_sets": {0: "all"}, "max_paths": 25},
    trim_params={"idxs": np.linspace(0, 497930, 6000).astype(int)},
    ch_params=p,
)
```

TX set 4 (`TX_3`), RX set 0 (`RX_1`). `trim_params['idxs']` is applied **before**
channels are computed, so only 6,000 of the 497,931 users in the set are ever
evaluated — this is the single biggest time saving in the pipeline (14 s total).

**Confirmed: `trim_params['idxs']` indexes within the selected RX set, not
globally across all three.** `n_ue` came back as exactly 6,000.

## 2. Antenna and subcarrier counts actually used

| | |
|---|---|
| BS antennas | **64** (ULA, `[64,1]`) |
| UE antennas | 1 |
| Subcarriers retained | **32**, as `linspace(0, 511, 32)` |
| FFT size | 512 (10 MHz band) |
| Paths | 25 |

The 32 subcarriers are **spread across the full 512-bin / 10 MHz band** rather
than taken as contiguous bins 0–31, which would be a narrow, highly correlated
slice leaving the subcarrier axis nearly flat. **State this choice in the paper.**

## 3. User counts

| | |
|---|---|
| Receivers in scenario | 1,184,941 (18 transmitters) |
| Users in RX set 0 (`RX_1`) | 497,931 |
| Channels generated (subsampled) | **6,000** |
| Dropped as invalid (dead users) | **0** |
| Retained | **6,000** |
| Used (500 seed + 2000 test) | 2,500 |

**Zero dead users.** Filtering was `isfinite(H).all()` AND
`‖H‖_F > 0`; neither dropped anything. This is consistent with
`deepmimo.summary('o1_28')`, where every RX set reports *Active points == Total
points*. The paper should report "0 of 6,000 sampled users were discarded" and
note that `o1_28`'s receiver grids contain no blocked/out-of-coverage points, so
the filter is a safeguard rather than a significant preprocessing step.

## 4. Position normalization bounds

```
x: [242.423, 278.423]   (span  36.0 m)
y: [297.171, 847.171]   (span 550.0 m)
```

Normalized to `[-1, 1]` per axis over the **full retained set** (all 6,000),
via `2*(pos - lo)/(hi - lo) - 1`. z is dropped (O1 receivers are ground level).

**Phases 4/5/7 must condition the GAN with this identical transform.**

Note the geometry: the user set is a **long, narrow strip** — 550 m along y but
only 36 m across x. This is what drives the skewed beam distribution recorded in
`HANDOFF_PHASE_2.md`, and the paper should describe the user region explicitly.

## 5. Normalization convention

**Per-sample Frobenius norm**, confirmed. Each sample is divided by its own
‖·‖_F over the full `[2,64,32]` tensor, and the divisors are **saved** as
`seed_norm.pt` / `test_norm.pt` so spectral efficiency can be computed on
unnormalized channels.

## 6–7. Saved artifacts, shapes and dtypes

All under `/kaggle/working/data/`:

| File | Shape | dtype |
|---|---|---|
| `seed_csi.pt` | `[500, 2, 64, 32]` | `torch.float32` |
| `test_csi.pt` | `[2000, 2, 64, 32]` | `torch.float32` |
| `seed_pos.pt` | `[500, 2]` | `torch.float32` |
| `test_pos.pt` | `[2000, 2]` | `torch.float32` |
| `seed_norm.pt` | `[500]` | `torch.float32` |
| `test_norm.pt` | `[2000]` | `torch.float32` |
| `seed_idx.npy` | `(500,)` | int64 |
| `test_idx.npy` | `(2000,)` | int64 |

Split: `np.random.RandomState(42).choice(6000, 2500, replace=False)`, first 500
seed, remaining 2000 test. Disjointness asserted at runtime.

Checkpoint assertions that passed: both CSI shapes exact, all values finite,
`seed_pos` within `[-1, 1]`.

## The raw channel axis order (was the last unverified fact)

```
raw channel shape: (6000, 1, 64, 32) complex64
      -> [n_users, n_rx_antennas, n_tx_antennas, n_subcarriers]
oriented channel shape: (6000, 64, 32)
```

`ds.ch` is `complex64`. `ds.rx_pos` holds `(x, y, z)`; only x and y are kept.
The dataset attribute names in use are `ch` and `rx_pos` (resolved and printed
at runtime by `_first_attr()`).

`generate()` with a single TX set and a single RX set returns a **plain
`Dataset`, not a `MacroDataset`** — there is no `.datasets` list to index.
