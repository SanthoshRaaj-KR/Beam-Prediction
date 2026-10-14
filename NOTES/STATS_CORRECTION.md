# Statistics correction — the sweep's real finding

**Date:** 2026-10-14. Applies to the first *valid* sweep (after the generator
norm fix, `aa60a5a`). Supersedes the headline `run_all.py` printed for that run.

## What the pipeline printed, and why it was wrong

```
best_ratio 50 | improvement_pp 0.65 | mannwhitney_p 0.5066 | ttest_p 0.5336
```

`significance()` only ever tested **best ratio vs R0** — the single weakest
comparison available, at n=3 per group. It reported "no effect" and in doing so
hid the sweep's actual, significant result: a **decline at high ratios**.
`significance()` has been rewritten to report three tests in priority order
(trend → pooled low-vs-high → legacy best-vs-R0).

## The real numbers

All values below are computed directly from the saved
`Result/ablation_results.csv` (21 rows, committed to this repo), **not** from
the run log. The log paste was missing the R50 seed-0 block and mislabelled one
R25 row, so anything derived from the log alone is unreliable; the CSV is the
only source to quote from.

| Ratio | Top-1 mean ± std | Top-3 mean ± std |
|---|---|---|
| 0 | 0.7823 ± 0.0133 | 0.9317 ± 0.0064 |
| 25 | 0.7847 ± 0.0090 | 0.9410 ± 0.0065 |
| 50 | **0.7888 ± 0.0096** | 0.9380 ± 0.0048 |
| 75 | 0.7813 ± 0.0189 | 0.9343 ± 0.0064 |
| 100 | 0.7655 ± 0.0064 | 0.9333 ± 0.0100 |
| 150 | 0.7743 ± 0.0193 | 0.9355 ± 0.0030 |
| 200 | 0.7658 ± 0.0093 | 0.9322 ± 0.0114 |

### 1. Trend over all 21 runs — PRIMARY

| | |
|---|---|
| Spearman ρ | **−0.437, p = 0.0478** |
| OLS slope | **−1.01 pp per 100% synthetic added, p = 0.0276** |

A single test of the paper's actual question ("does the ratio matter"),
specifiable before seeing the curve. This is the number to lead with.

### 2. Pooled low (25/50) vs high (100/150/200) — SECONDARY, post-hoc

Split at the observed peak R50: low = {25, 50} (n=6), high = {75, 100, 150,
200} (n=12).

| | |
|---|---|
| low mean | 0.7867 |
| high mean | 0.7717 |
| Δ | **+1.50 pp** |
| Welch | **p = 0.0142** |
| Mann-Whitney | **p = 0.0245** |
| Cohen's d | **1.27** |

The split sits at the observed peak, so it was chosen after seeing the data.
**Must be labelled exploratory in the paper.** Note this grouping puts R75 in
the "high" arm, which is the less flattering and therefore more defensible
choice -- excluding R75 raises Δ to +1.82 pp (p = 0.0047) but is more clearly
cherry-picked.

### 3. Pairwise (Welch) — supporting

| Comparison | Δ | p |
|---|---|---|
| R50 vs R100 | −2.33 pp | 0.031 |
| R50 vs R200 | −2.30 pp | 0.041 |
| R25 vs R200 | −1.88 pp | 0.066 |
| R0 vs R100 | −1.68 pp | 0.146 |
| R0 vs R50 (old headline) | +0.65 pp | 0.53 — **not** significant |

Note: Mann-Whitney's minimum two-sided p at 3 vs 3 is **0.1**, so it cannot
resolve any pairwise effect in this design. Quote Welch and state the reason.
None of the pairwise tests survives Bonferroni over 7 comparisons (α = 0.007),
which is another reason to lead with the single trend test.

## The honest headline

> Adding GAN-synthesized CSI beyond roughly 50% of the real seed set
> significantly degrades Top-1 beam-prediction accuracy (−1.01 pp per 100%
> synthetic added, p = 0.028 over 21 runs). Modest ratios are neutral at best:
> the apparent +0.65 pp peak at 50% is not statistically resolvable (p = 0.53).
> The "small synthetic fraction is best" behaviour reported for GAN-augmented
> image classification therefore transfers *qualitatively* to mmWave beam
> prediction, but in this setting only the penalty is detectable, not the
> benefit.

## Mechanism — three independent signs of mode collapse

1. **Coverage.** Synthetic set populates **9 of 32** beam classes (real: 32/32).
   Adding it amplifies the existing class imbalance (majority class already
   53.84%).
2. **Top-1 falls, Top-3 does not.** R0 → R200: Top-1 **−1.65 pp**, Top-3
   **+0.05 pp**. Exactly what class-imbalance amplification predicts — rare
   beams get crushed, dominant beams are untouched.
3. **Training loss falls as data grows.** R150/R200 reach train loss
   0.000–0.008 versus R0's 0.006–0.058, despite 3× the samples. The synthetic
   samples are easy and redundant.

## Still open: discriminability 1.0000 ± 0.0000 (second time)

The norm fix worked on its target — power-profile Wasserstein is now 0.00006
and label overlap rose 0.54 → 0.774 — yet the GAN-test discriminator is still
perfect. The earlier heuristic ("zero variance ⇒ scalar artifact") does not
settle it. Leading hypothesis is **structural, not a bug**: ray-traced CSI at
max path depth 4 is nearly low-rank,
`H = Σ_{p≤4} α_p · steer(θ_p) · delay(τ_p)ᵀ` — at most four rank-1 outer
products of complex exponentials, i.e. an algebraic variety rather than the
diffuse manifold natural images occupy. A ConvTranspose generator emits
generically full-rank output, which a CNN detects instantly via local phase
coherence.

`evaluation/mechanism_check.py::rank_spectrum()` tests this (Figure 6). If real
spectra show a cliff after a few singular values and synthetic ones decay
slowly, the hypothesis holds and this becomes the paper's explanation for why
GAN augmentation underperforms on ray-traced CSI relative to imaging domains.
If the spectra overlap, the hypothesis is wrong and the separator is something
else.

`run_tstr()` is the companion check: train on 400 synthetic + 100 synthetic val
(mirroring R0's split), test on the same 2000 real samples. With 9/32 coverage
it should land near the 53.84% majority floor.
