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
    """Best ratio vs R0: Mann-Whitney U (n=3) plus the t-test, and the
    absolute improvement in percentage points (the abstract's headline).

    With 3 seeds, state plainly that power is limited and that differences
    within one standard deviation are NOT claimed as significant.
    """
    from scipy import stats

    r0 = df[df["synthetic_ratio"] == 0]["top1_acc"].values
    if best_ratio is None:
        means = df.groupby("synthetic_ratio")["top1_acc"].mean()
        best_ratio = int(means.idxmax())
    best = df[df["synthetic_ratio"] == best_ratio]["top1_acc"].values

    out = {"best_ratio": best_ratio,
           "r0_mean": float(r0.mean()), "best_mean": float(best.mean()),
           "improvement_pp": float(100 * (best.mean() - r0.mean()))}
    try:
        out["mannwhitney_p"] = float(stats.mannwhitneyu(best, r0, alternative="two-sided")[1])
    except Exception:
        out["mannwhitney_p"] = float("nan")
    try:
        out["ttest_p"] = float(stats.ttest_ind(best, r0, equal_var=False)[1])
    except Exception:
        out["ttest_p"] = float("nan")

    print("Best ratio R%d: %.4f vs R0 %.4f  ->  %+.2f pp" % (
        best_ratio, out["best_mean"], out["r0_mean"], out["improvement_pp"]))
    print("Mann-Whitney p = %.4f | Welch t-test p = %.4f  (n=3 per group -- "
          "limited power, do not claim significance within 1 std)" % (
              out["mannwhitney_p"], out["ttest_p"]))
    return out
