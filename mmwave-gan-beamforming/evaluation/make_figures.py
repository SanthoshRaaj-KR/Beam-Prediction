"""Phase 8 -- figures, tables and statistics. Maps 1:1 onto the paper's slots.

  fig_ablation_curve.png     Figure 5 (KEY)
  fig_gan_training_curve.png Figure 2
  fig_label_hist.png         Figure 1
  table_main_results.md      Table III
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PLOT_DIR = "/kaggle/working/results/plots"
TABLE_DIR = "/kaggle/working/results"


def fig_label_hist(labels, majority_pct, plot_dir=PLOT_DIR):
    """Figure 1 -- beam-label distribution over the retained user set.

    Sanity gate: more than ~10 of 32 classes must be populated. If 90% of
    samples land on 1-2 beams, the codebook math or array axis is wrong.
    """
    os.makedirs(plot_dir, exist_ok=True)
    lab = labels.numpy() if hasattr(labels, "numpy") else np.asarray(labels)
    counts = np.bincount(lab, minlength=32)
    populated = int((counts > 0).sum())

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(np.arange(32), counts, color="#1f77b4")
    ax.set_xlabel("Beam index"); ax.set_ylabel("Number of users")
    ax.set_title("Optimal-beam distribution (%d/32 classes populated, "
                 "majority class = %.1f%%)" % (populated, 100 * majority_pct))
    fig.tight_layout()
    path = os.path.join(plot_dir, "fig_label_hist.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("Figure 1 -> %s | %d/32 populated" % (path, populated))
    return {"plot": path, "n_populated": populated, "counts": counts.tolist()}


def fig_gan_training_curve(w_hist, plot_dir=PLOT_DIR, table_dir=None, smooth=25):
    """Figure 2 -- Wasserstein estimate vs generator iteration.

    Expect a generally decreasing, stabilizing curve. Oscillation is normal;
    a clean monotonic curve is not expected and its absence is not a bug.
    """
    os.makedirs(plot_dir, exist_ok=True)
    it = np.array([h["it"] for h in w_hist])
    w = np.array([h["w_est"] for h in w_hist])

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(it, w, alpha=0.3, color="#1f77b4", lw=0.8, label="per iteration")
    if len(w) >= smooth:
        k = np.ones(smooth) / smooth
        ax.plot(it[smooth - 1:], np.convolve(w, k, mode="valid"),
                color="#d62728", lw=1.8, label="moving avg (%d)" % smooth)
    ax.set_xlabel("Generator iteration")
    ax.set_ylabel("Wasserstein estimate  $D(x)-D(\\tilde{x})$")
    ax.set_title("WGAN-GP training")
    ax.legend(); fig.tight_layout()

    path = os.path.join(plot_dir, "fig_gan_training_curve.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    table_dir = TABLE_DIR if table_dir is None else table_dir
    csv_path = os.path.join(table_dir, "gan_training_curve.csv")
    os.makedirs(table_dir, exist_ok=True)
    with open(csv_path, "w") as f:
        f.write("it,w_est,d_loss,g_loss\n")
        for h in w_hist:
            f.write("%d,%.6f,%.6f,%.6f\n" % (h["it"], h["w_est"], h["d_loss"], h["g_loss"]))
    print("Figure 2 -> %s (+ %s)" % (path, csv_path))
    return {"plot": path, "csv": csv_path}


def fig_ablation_curve(agg, majority_pct, plot_dir=PLOT_DIR):
    """Figure 5 -- THE key figure. Top-1 and Top-3 vs synthetic ratio."""
    os.makedirs(plot_dir, exist_ok=True)
    r = agg["synthetic_ratio"].values
    t1m, t1s = agg["top1_mean"].values, np.nan_to_num(agg["top1_std"].values)
    t3m, t3s = agg["top3_mean"].values, np.nan_to_num(agg["top3_std"].values)

    fig, ax = plt.subplots(figsize=(7, 4.6))
    ax.errorbar(r, 100 * t1m, yerr=100 * t1s, marker="o", capsize=4,
                color="#1f77b4", label="Top-1")
    ax.errorbar(r, 100 * t3m, yerr=100 * t3s, marker="s", capsize=4,
                color="#2ca02c", label="Top-3", ls="--")
    ax.axhline(100 * majority_pct, ls=":", color="grey",
               label="Majority-class baseline (%.1f%%)" % (100 * majority_pct))
    ax.set_xlabel("Synthetic-to-real ratio (%)")
    ax.set_ylabel("Beam-prediction accuracy (%)")
    ax.set_title("Beam-prediction accuracy vs. synthetic data ratio")
    ax.set_xticks(r)
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(plot_dir, "fig_ablation_curve.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("Figure 5 -> %s" % path)
    return path


def table_main_results(agg, majority_pct, table_dir=TABLE_DIR):
    """Table III -- ratio x {Top-1, Top-3, SE%} as mean +/- std."""
    os.makedirs(table_dir, exist_ok=True)
    lines = [
        "| Run | Ratio | Top-1 Acc (%) | Top-3 Acc (%) | Spectral Eff. (%) |",
        "|---|---|---|---|---|",
        "| Majority-class | - | %.2f | - | - |" % (100 * majority_pct),
    ]
    for _, row in agg.iterrows():
        lines.append("| R%d | %d%% | %.2f ± %.2f | %.2f ± %.2f | %.2f ± %.2f |" % (
            row["synthetic_ratio"], row["synthetic_ratio"],
            100 * row["top1_mean"], 100 * np.nan_to_num(row["top1_std"]),
            100 * row["top3_mean"], 100 * np.nan_to_num(row["top3_std"]),
            row["se_mean"], np.nan_to_num(row["se_std"])))
    md = "\n".join(lines)
    path = os.path.join(table_dir, "table_main_results.md")
    with open(path, "w") as f:
        f.write(md + "\n")
    print("Table III -> %s\n" % path)
    print(md)
    return path


def significance(df, best_ratio=None):
    """Statistics for the ratio sweep, ordered by how much weight they bear.

    This ordering is the point. The first valid sweep printed `best ratio vs
    R0` as its headline -- the WEAKEST comparison in the data (+0.65 pp,
    p=0.53) -- while the sweep's actual finding, a significant decline at high
    ratios, went unreported. Lead with the trend test.

      1. TREND over all runs (PRIMARY). One test of the paper's actual
         question -- does the synthetic ratio matter -- and specifiable before
         looking at the curve. Spearman rho plus an OLS slope in pp per 100%
         synthetic added.
      2. LOW vs HIGH pooled (SECONDARY, EXPLORATORY). Split at the observed
         peak, therefore chosen after seeing the data. More samples per group
         than any pairwise test, but it must be labelled post-hoc in the paper.
      3. BEST vs R0 (LEGACY). Kept for continuity only. n=3 per group has
         almost no power; do not lead with it.

    Mann-Whitney cannot return p < 0.1 for 3 vs 3 -- 0.1 is its minimum
    two-sided p at that size -- so Welch is quoted for the pairwise tests and
    the reason is printed rather than left for a reviewer to find.
    """
    from scipy import stats

    means = df.groupby("synthetic_ratio")["top1_acc"].mean()
    if best_ratio is None:
        best_ratio = int(means.idxmax())

    r0 = df[df["synthetic_ratio"] == 0]["top1_acc"].values
    best = df[df["synthetic_ratio"] == best_ratio]["top1_acc"].values
    out = {"best_ratio": best_ratio,
           "r0_mean": float(r0.mean()), "best_mean": float(best.mean()),
           "improvement_pp": float(100 * (best.mean() - r0.mean()))}

    # ---- 1. trend across every run (primary) ------------------------------
    x = df["synthetic_ratio"].values.astype(float)
    y = df["top1_acc"].values.astype(float)
    rho, p_rho = stats.spearmanr(x, y)
    slope, _, _, p_slope, _ = stats.linregress(x, y)
    out["trend_spearman_rho"] = float(rho)
    out["trend_spearman_p"] = float(p_rho)
    out["trend_slope_pp_per_100"] = float(slope * 100 * 100)
    out["trend_slope_p"] = float(p_slope)

    # ---- 2. pooled low vs high, split at the peak (exploratory) -----------
    low = df[(df["synthetic_ratio"] > 0)
             & (df["synthetic_ratio"] <= best_ratio)]["top1_acc"].values
    high = df[df["synthetic_ratio"] > best_ratio]["top1_acc"].values
    out["low_ratio_mean"] = float(low.mean()) if len(low) else float("nan")
    out["high_ratio_mean"] = float(high.mean()) if len(high) else float("nan")
    out["low_vs_high_pp"] = float("nan")
    out["low_vs_high_welch_p"] = float("nan")
    out["low_vs_high_mw_p"] = float("nan")
    out["low_vs_high_cohen_d"] = float("nan")
    if len(low) > 1 and len(high) > 1:
        out["low_vs_high_pp"] = float(100 * (low.mean() - high.mean()))
        out["low_vs_high_welch_p"] = float(
            stats.ttest_ind(low, high, equal_var=False)[1])
        out["low_vs_high_mw_p"] = float(
            stats.mannwhitneyu(low, high, alternative="two-sided")[1])
        pooled = np.sqrt((low.var(ddof=1) + high.var(ddof=1)) / 2.0)
        out["low_vs_high_cohen_d"] = float(
            (low.mean() - high.mean()) / max(pooled, 1e-12))

    # ---- 3. best vs R0 (legacy) ------------------------------------------
    try:
        out["mannwhitney_p"] = float(stats.mannwhitneyu(best, r0, alternative="two-sided")[1])
    except Exception:
        out["mannwhitney_p"] = float("nan")
    try:
        out["ttest_p"] = float(stats.ttest_ind(best, r0, equal_var=False)[1])
    except Exception:
        out["ttest_p"] = float("nan")

    print()
    print("[1] PRIMARY -- trend over all %d runs" % len(df))
    print("    Spearman rho = %+.3f (p = %.4f)" % (rho, p_rho))
    print("    OLS slope    = %+.2f pp per 100%% synthetic added (p = %.4f)"
          % (out["trend_slope_pp_per_100"], p_slope))
    print("[2] SECONDARY (post-hoc, split at the observed peak R%d)" % best_ratio)
    print("    ratios 1-%d%% n=%d mean %.4f  vs  >%d%% n=%d mean %.4f"
          % (best_ratio, len(low), low.mean() if len(low) else float("nan"),
             best_ratio, len(high), high.mean() if len(high) else float("nan")))
    print("    delta = %+.2f pp | Welch p = %.4f | MW p = %.4f | Cohen d = %.2f"
          % (out["low_vs_high_pp"], out["low_vs_high_welch_p"],
             out["low_vs_high_mw_p"], out["low_vs_high_cohen_d"]))
    print("[3] LEGACY -- best R%d %.4f vs R0 %.4f -> %+.2f pp"
          % (best_ratio, out["best_mean"], out["r0_mean"], out["improvement_pp"]))
    print("    Welch p = %.4f | MW p = %.4f" % (out["ttest_p"], out["mannwhitney_p"]))
    print("    n=3 per group. Mann-Whitney's floor at 3 vs 3 is p=0.1, so MW")
    print("    cannot resolve any pairwise effect here -- quote Welch and say so.")
    print("    Do NOT headline [3]; it is the least powered test in the sweep.")
    return out
