# Kaggle run order — paste these cells in sequence

Upload the whole `mmwave-gan-beamforming/` folder as a **Kaggle Dataset** (or
`!git clone` it), then `sys.path.insert(0, <repo root>)`. Every cell below
assumes the repo root is importable.

**Save `/kaggle/working/` to a Kaggle Dataset before the 7-hour session dies.**
You will cross that boundary somewhere around the GAN training cell.

---

## Cell 1 — install
```python
!pip install DeepMIMO
# >>> Run > Restart Session NOW, then start at Cell 2 <<<
```

## Cell 2 — environment check + download (Phase 0)
```python
import deepmimo, os, torch, sys
print(deepmimo.__version__, torch.cuda.is_available())   # 4.0.5 observed
!df -h / /kaggle/working

os.makedirs('/root/deepmimo_scenarios', exist_ok=True)
deepmimo.download('o1_28', output_dir='/root/deepmimo_scenarios')
```

**`output_dir` does NOT control the final destination.** The zip downloads and
extracts there, then `shutil.move()` targets the library's configured scenarios
dir — `/kaggle/working/deepmimo_scenarios/` — which is a different device, so it
falls back to `copytree` and dies against the 20 GB cap with
`Invalid cross-device link` followed by `No space left on device`.

**The download and extract still succeeded.** Recover with Cell 2b rather than
re-downloading (re-downloading risks the HTTP 429 rate limit for no gain).

## Cell 2b — recover from the cross-device move failure
```python
# 1. free the quota -- the partial copy filled /kaggle/working
!rm -rf /kaggle/working/deepmimo_scenarios
!df -h /kaggle/working

# 2. the source survives (copytree raises before rmtree(src) runs)
!ls /root/deepmimo_scenarios/o1_28 | wc -l        # expect 4161

# 3. symlink instead of copying -- zero bytes moved
import os, deepmimo
os.makedirs('/kaggle/working/deepmimo_scenarios', exist_ok=True)
os.symlink('/root/deepmimo_scenarios/o1_28',
           '/kaggle/working/deepmimo_scenarios/o1_28')
print(deepmimo.get_available_scenarios())          # should list o1_28
```
If the symlink is not enough, repoint the scenarios directory itself — inspect
`deepmimo.config` and set it to `/root/deepmimo_scenarios`, keeping raw data off
`/kaggle/working` entirely.

## Cell 3 — map the API (this is the Phase 0 handoff)
```python
help(deepmimo.load); help(deepmimo.generate); help(deepmimo.ChannelParameters)
print([x for x in dir(deepmimo) if not x.startswith('_')])
help(deepmimo.steering_vec)      # may replace the hand-rolled codebook
```
Capture the **full output** — it is what `data/preprocess.py:load_raw()` needs.

## Cell 4 — smoke test (no data needed; run it while the download finishes)
```python
sys.path.insert(0, '/kaggle/input/<your-dataset>/mmwave-gan-beamforming')
!cd /kaggle/input/<your-dataset>/mmwave-gan-beamforming && python smoke_test.py
```
Must print `ALL SMOKE TESTS PASSED`. If it does not, fix that before spending
GPU time on real data.

## Cell 5 — Phase 1: preprocessing
`load_raw()` is written and configured for DeepMIMO 4.0.5 — just run it:
```python
from data.preprocess import load_raw, preprocess
H_all, pos_all = load_raw()        # BS set 4 (TX_3), RX set 0, 6000 UEs
info = preprocess(H_all, pos_all)
print(info)        # record n_dropped / n_retained / pos bounds -> HANDOFF_PHASE_1
```
It subsets 6,000 UEs **before** computing channels (via `trim_params['idxs']`),
so you never pay for the ~498k users you don't use. 6,000 leaves margin over
the 2,500 needed after dead-user filtering.

Config applied: `bs_antenna.shape=[64,1]` (the default is `[8,1]` — setting
this is mandatory), `ue_antenna.shape=[1,1]`, and 32 subcarriers selected as
`linspace(0, 511, 32)` — spread across the full 10 MHz band rather than taking
bins 0–31, which would be a narrow, highly correlated slice. State that choice
in the paper.

**Two things to check in the output:**
- The printed `raw channel shape` — record it for the handoff.
- `n_retained` must be ≥ 2500. If it isn't, raise `N_SUBSAMPLE` in
  `data/preprocess.py` and re-run.

If the UE count comes back as something other than 6,000, `trim_params['idxs']`
is probably indexing globally across all three RX sets rather than within set 0.

## Cell 6 — Phase 2: codebook + labels
```python
import torch
from utils.codebook import build_codebook, label_batch, majority_class_accuracy
from evaluation.make_figures import fig_label_hist

D = '/kaggle/working/data'
seed_csi, test_csi = torch.load(f'{D}/seed_csi.pt'), torch.load(f'{D}/test_csi.pt')
seed_pos, test_pos = torch.load(f'{D}/seed_pos.pt'), torch.load(f'{D}/test_pos.pt')
seed_norm, test_norm = torch.load(f'{D}/seed_norm.pt'), torch.load(f'{D}/test_norm.pt')

cb = build_codebook()
seed_labels, test_labels = label_batch(seed_csi, cb), label_batch(test_csi, cb)
torch.save(cb, f'{D}/codebook.pt')
torch.save(seed_labels, f'{D}/seed_labels.pt')
torch.save(test_labels, f'{D}/test_labels.pt')

maj, maj_cls = majority_class_accuracy(torch.cat([seed_labels, test_labels]))
print('MAJORITY-CLASS BASELINE:', round(maj, 4), 'class', maj_cls)
fig_label_hist(torch.cat([seed_labels, test_labels]), maj)
```
**Gate:** more than ~10 of 32 classes populated. If 90% of samples land on 1–2
beams, debug the array axis / conjugation before going further.

## Cell 7 — Phase 3 + 4: R0 baseline and shape checks
```python
from evaluation.run_ablation import run_sweep
r0 = run_sweep(seed_csi, seed_labels, test_csi, test_labels, cb,
               test_norms=test_norm, ratios=[0], seeds=[0, 1, 2])
```
**THE GATE:** R0 Top-1 must clearly beat the majority-class baseline. If it does
not, stop and debug the labels, data, or training loop — no GAN work fixes it.

## Cell 8 — Phase 5: GAN training (the long one)
```python
from training.train_gan import train_gan
G, Dc, w_hist = train_gan(seed_csi, seed_pos, n_iters=3000, seed=42)
```
Resumes automatically from the newest `/kaggle/working/checkpoints/gan_*.pt`.
Watch the Wasserstein estimate — it should generally decrease and stabilize.
Oscillation is normal; `d_loss`/`g_loss` raw values mean nothing in WGAN. A NaN
is a bug, not bad luck.

**Save `/kaggle/working/` to a Kaggle Dataset here.**

## Cell 9 — Phase 6: quality check
```python
from training.train_gan import generate_synthetic
from evaluation import gan_quality_check as qc

synth, _ = generate_synthetic(G, seed_pos, 500, seed=0)
qc.test1_heatmaps(seed_csi, synth)
qc.test2_power_profile(seed_csi, synth)
t3 = qc.test3_discriminability(seed_csi, synth)        # headline number
t4 = qc.test4_label_consistency(seed_labels, synth, cb)
```
Record `t3['mean'] ± t3['std']` and `t4['overlap']`. A high discriminability
number is a legitimate finding — it *explains* weak augmentation gains. Report
it, do not hide it.

## Cell 10 — Phase 7: the sweep (18 remaining runs)
```python
rows = run_sweep(seed_csi, seed_labels, test_csi, test_labels, cb,
                 G=G, seed_pos=seed_pos, test_norms=test_norm,
                 ratios=[25, 50, 75, 100, 150, 200], seeds=[0, 1, 2])
```
Behind on time? Cut **ratios, not seeds** — `[50, 100, 200]` × 3 seeds beats 7
ratios × 1 seed, because error bars are what make the claim defensible.

## Cell 11 — Phase 8: figures, table, statistics
```python
from evaluation.run_ablation import aggregate
from evaluation import make_figures as mf

df, agg = aggregate()
mf.fig_gan_training_curve(w_hist)
mf.fig_ablation_curve(agg, maj)
mf.table_main_results(agg, maj)
sig = mf.significance(df)
print(sig)        # improvement_pp is the abstract's headline number
```

## Cell 12 — persist everything
```python
!ls -R /kaggle/working/results /kaggle/working/checkpoints | head -50
```
Commit the notebook / save the Kaggle Dataset. Then fill `PAPER_DRAFT.md` —
73 placeholders, all numeric, sources listed in its Appendix A.
