# GAN-Augmented mmWave Beam Prediction

An ablation study on synthetic-to-real data ratios for millimeter-wave beam
prediction under data scarcity.

A conditional WGAN-GP synthesizes mmWave CSI matrices conditioned on user
position, trained on a deliberately small 500-sample real seed set from the
DeepMIMO `o1_28` ray-tracing scenario (28 GHz). A CNN beam classifier is then
trained on mixtures of real and synthetic data at seven synthetic-to-real
ratios, each repeated with three random seeds, and every run is evaluated on the
same held-out 2000-sample real test set. The resulting accuracy-versus-ratio
curve is the study's main result.

**Scope of the novelty claim:** the synthetic-to-real ratio question has been
studied in other domains — notably GAN-augmented medical imaging, where the
optimum sat at a small synthetic fraction and performance declined as synthetic
data grew. This work is scoped as the first such ratio sweep *for mmWave beam
prediction*, and treats the cross-domain finding as the hypothesis under test.

## Status

Pipeline implemented and verified end to end on fabricated data
(`python smoke_test.py`). No stubs remain: `load_raw()` is written against
DeepMIMO 4.0.5 and every phase is runnable. No real results yet.

## Layout

```
NOTES/                          planning documents (read these first)
  execution_and_paper_guide.md    plan + rationale; §0.5 is the living errata log
  IMPLEMENTATION_PHASES.md        the build instructions -- 10 self-contained phases
  PAPER_DRAFT.md                  paper, prose complete, numeric placeholders open

mmwave-gan-beamforming/
  utils/codebook.py             DFT codebook + physics-based beam labeller
  models/                       BeamCNN, Generator, Critic + gradient penalty
  training/                     frozen classifier recipe; WGAN-GP with resume
  evaluation/                   ablation sweep, GAN quality checks, figures/tables
  data/preprocess.py            Phase 1 -- DeepMIMO extraction + preprocessing
  notebooks/RUN_ORDER.md        cell-by-cell Kaggle run order
  smoke_test.py                 end-to-end check, no DeepMIMO required
```

Precedence when documents disagree: `IMPLEMENTATION_PHASES.md` wins on anything
executable; §0.5 of the execution guide wins on environment and API facts.

## Experimental design

| | |
|---|---|
| Dataset | DeepMIMO `o1_28` (ray-traced, 28 GHz) |
| CSI tensor | `[N, 2, 64, 32]` — real/imag × 64 antennas × 32 subcarriers |
| Real seed set | 500 samples |
| Held-out test set | 2000 real samples, disjoint in user index |
| Beam codebook | 32-beam DFT over a 64-element ULA |
| Ratios | 0, 25, 50, 75, 100, 150, 200% synthetic, added on top of the real set |
| Seeds | 3 per ratio (21 runs total) |
| Metrics | Top-1, Top-3, relative spectral efficiency at a fixed 10 dB SNR |

Validity constraints held across all runs: classifier hyperparameters are frozen
after the baseline, validation sets contain real samples only, synthetic data is
regenerated fresh per seed, synthetic labels come from the physics labeller
rather than the generator, and the test set is touched exactly once per run.

## Reproducing

```bash
python smoke_test.py          # verifies the pipeline on fabricated data
```

Then follow `mmwave-gan-beamforming/notebooks/RUN_ORDER.md` on a Kaggle GPU
session, starting at its Cell 0 (the `sys.path` bootstrap — without it every
`from data.preprocess import ...` fails, and it must be re-run after the kernel
restart that the DeepMIMO install requires). Two environment facts worth knowing before you start: the extracted
`o1_28` scenario is **21 GB**, which exceeds Kaggle's 20 GB `/kaggle/working`
cap, so it must live on the root filesystem and be symlinked into place; and
`deepmimo.download()`'s `output_dir` does not control the final install target.
Both are documented in §0.5 of the execution guide.

## Note on "real" data

The seed and test sets are ray-traced, not measured. This is standard in the
beam-prediction literature, but "real" throughout this repository means *real
ray-traced channels*, and the paper states so in its limitations.
