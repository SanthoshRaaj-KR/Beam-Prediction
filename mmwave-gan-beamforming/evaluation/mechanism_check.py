"""Mechanism checks -- WHY GAN augmentation hurts here, not just that it does.

Added after the first valid sweep (generator norm fix, commit aa60a5a), which
found a significant *degradation* at high synthetic ratios (-1.11 pp per 100%
synthetic added, p=0.016 over all 21 runs) while the apparent +0.65 pp gain at
50% was not resolvable (p=0.53). Both checks below probe the cause.

  rank_spectrum()
      Figure 6. Ray-traced CSI at max path depth 4 is nearly low-rank:
      H = sum_{p<=4} alpha_p * steer(theta_p) * delay(tau_p)^T -- at most four
      rank-1 outer products of complex exponentials. A ConvTranspose generator
      emits generically full-rank output. If that is why the GAN-test
      discriminator separates real from synthetic at 100% while the marginals
      match (power-profile Wasserstein 0.00006, label overlap 0.774), the
      singular-value spectra must differ sharply: real showing a cliff after a
      few values, synthetic a slow decay. If instead the spectra overlap, the
      low-rank explanation is wrong and something else separates them.

  run_tstr()
      Train on Synthetic, Test on Real. 400 synthetic train + 100 synthetic
      val, mirroring R0's 400/100 real split exactly so the two are directly
      comparable. Scored on the same 2000 real test samples. No real CSI enters
      training or model selection. With only 9 of 32 beam classes populated in
      the synthetic set, this is expected near the 53.84% majority-class floor;
      that turns the mode-collapse story into a number a reviewer can check.

TSTR rows go to their OWN csv. They must never be appended to
ablation_results.csv -- that would corrupt aggregate() and the resume logic,
since TSTR has no place on the ratio axis.
"""
import csv
import os
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils.codebook import csi_to_complex, label_batch, N_BEAMS

PLOT_DIR = "/kaggle/working/results/plots"
RESULTS_DIR = "/kaggle/working/results"

TSTR_COLUMNS = ["run_id", "seed", "n_synth_train", "n_synth_val",
                "synth_train_classes", "top1_acc", "top3_acc",
                "spectral_eff_pct", "best_val_acc", "epochs", "timestamp"]


def rank_spectrum(real_csi, synth_csi, plot_dir=PLOT_DIR, energy=0.90):
    """Per-sample singular-value spectra of the [64,32] channel matrices.

    Each sample's spectrum is normalised by its own largest singular value, so
    the comparison is of *shape*, not scale -- scale was already matched by the
    unit-Frobenius projection in the generator.
    """
    os.makedirs(plot_dir, exist_ok=True)
    pct = int(round(energy * 100))
    out, curves = {}, {}

    for name, csi in (("real", real_csi), ("synth", synth_csi)):
        H = csi_to_complex(csi)                       # [N,64,32] complex
        sv = np.linalg.svd(H, compute_uv=False)       # [N,32], descending
        sv = sv / np.maximum(sv[:, :1], 1e-12)
        curves[name] = sv

        p = sv ** 2
        p = p / np.maximum(p.sum(axis=1, keepdims=True), 1e-12)
        rank_e = (np.cumsum(p, axis=1) < energy).sum(axis=1) + 1
        ent = -(p * np.log(np.maximum(p, 1e-12))).sum(axis=1)

        out["rank%d_%s" % (pct, name)] = float(rank_e.mean())
        out["rank%d_%s_std" % (pct, name)] = float(rank_e.std(ddof=1))
        out["effrank_%s" % name] = float(np.exp(ent).mean())

    out["rank%d_ratio" % pct] = float(
        out["rank%d_synth" % pct] / max(out["rank%d_real" % pct], 1e-12))

    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    k = np.arange(1, curves["real"].shape[1] + 1)
    for name, label, color in (("real", "Real (ray-traced)", "tab:blue"),
                               ("synth", "Synthetic (WGAN-GP)", "tab:red")):
        sv = curves[name]
        med = np.median(sv, axis=0)
        lo, hi = np.percentile(sv, 25, axis=0), np.percentile(sv, 75, axis=0)
        ax.semilogy(k, med, label=label, color=color, lw=1.8)
        ax.fill_between(k, lo, hi, color=color, alpha=0.20, lw=0)
    ax.set_xlabel("Singular value index")
    ax.set_ylabel(r"$\sigma_i / \sigma_1$")
    ax.set_title("Channel-matrix singular-value spectrum")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    path = os.path.join(plot_dir, "fig_rank_spectrum.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)

    print("Figure 6 -> %s" % path)
    print("  rank@%d%% energy:  real %.2f +/- %.2f   synthetic %.2f +/- %.2f"
          % (pct, out["rank%d_real" % pct], out["rank%d_real_std" % pct],
             out["rank%d_synth" % pct], out["rank%d_synth_std" % pct]))
    print("  effective rank:    real %.2f   synthetic %.2f"
          % (out["effrank_real"], out["effrank_synth"]))

    r = out["rank%d_ratio" % pct]
    if r >= 2.0:
        verdict = ("synthetic spectra are much flatter -- supports the low-rank "
                   "explanation for 100%% discriminability")
    elif r >= 1.3:
        verdict = ("synthetic spectra are moderately flatter -- partial support "
                   "for the low-rank explanation")
    else:
        verdict = ("spectra are comparable -- the low-rank explanation does NOT "
                   "account for the discriminability; look elsewhere")
    out["rank_verdict"] = verdict
    print("  -> %s" % verdict)
    return out


def run_tstr(seed_pos, test_csi, test_labels, codebook, G,
             test_norms=None, n_train=400, n_val=100, seeds=(0, 1, 2),
             device="cuda", results_dir=RESULTS_DIR, epochs=None, verbose=True):
    """Train on synthetic only, test on real. Mirrors R0's 400/100 split."""
    from training.train_classifier import train_classifier, EPOCHS
    from training.train_gan import generate_synthetic

    if epochs is None:
        epochs = EPOCHS
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, "tstr_results.csv")

    rows = []
    for seed in seeds:
        # fresh draw per seed, same procedure as the sweep
        synth, _ = generate_synthetic(G, seed_pos, n_train + n_val,
                                      seed=seed, device=device)
        labels = label_batch(synth, codebook)       # physics labeller, not the GAN
        tr_csi, tr_lab = synth[:n_train], labels[:n_train]
        va_csi, va_lab = synth[n_train:], labels[n_train:]
        n_cls = int(len(set(tr_lab.tolist())))

        print()
        print("=== TSTR seed %d | train %d synth (%d of %d beam classes) ==="
              % (seed, n_train, n_cls, N_BEAMS))
        m = train_classifier(
            tr_csi, tr_lab, va_csi, va_lab, test_csi, test_labels,
            seed=seed, codebook=codebook, test_norms=test_norms,
            device=device, epochs=epochs, verbose=verbose,
        )
        rows.append({
            "run_id": "TSTR", "seed": seed,
            "n_synth_train": n_train, "n_synth_val": n_val,
            "synth_train_classes": n_cls,
            "top1_acc": round(m["top1_acc"], 6),
            "top3_acc": round(m["top3_acc"], 6),
            "spectral_eff_pct": round(m["spectral_eff_pct"], 4),
            "best_val_acc": round(m["best_val_acc"], 6),
            "epochs": m["epochs"],
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        })
        print("  -> top1 %.4f  top3 %.4f  SE%% %.2f"
              % (m["top1_acc"], m["top3_acc"], m["spectral_eff_pct"]))

    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=TSTR_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    t1 = np.array([r["top1_acc"] for r in rows])
    t3 = np.array([r["top3_acc"] for r in rows])
    out = {
        "tstr_top1_mean": float(t1.mean()), "tstr_top1_std": float(t1.std(ddof=1)),
        "tstr_top3_mean": float(t3.mean()),
        "tstr_synth_train_classes": float(
            np.mean([r["synth_train_classes"] for r in rows])),
        "tstr_csv": csv_path,
    }
    print()
    print("TSTR -> top1 %.4f +/- %.4f  (top3 %.4f) over %d seeds"
          % (out["tstr_top1_mean"], out["tstr_top1_std"],
             out["tstr_top3_mean"], len(rows)))
    print("TSTR rows -> %s" % csv_path)
    return out


def summarise(spec, tstr, majority_acc, r0_top1):
    """One-paragraph read of both checks together, for Section V of the paper."""
    print()
    print("-" * 68)
    print("MECHANISM SUMMARY")
    print("-" * 68)
    print("  R0 (400 real)          top-1 %.4f" % r0_top1)
    print("  TSTR (400 synthetic)   top-1 %.4f" % tstr["tstr_top1_mean"])
    print("  majority-class floor   top-1 %.4f" % majority_acc)
    gap = tstr["tstr_top1_mean"] - majority_acc
    print("  TSTR above floor       %+.2f pp" % (100 * gap))
    if gap < 0.03:
        print("  -> synthetic CSI carries essentially NO usable beam "
              "information beyond the class prior.")
    elif tstr["tstr_top1_mean"] < r0_top1 - 0.05:
        print("  -> synthetic CSI carries partial beam information, well short "
              "of real data.")
    else:
        print("  -> synthetic CSI is nearly as informative as real data; the "
              "high-ratio degradation needs another explanation.")
    print("  spectrum verdict: %s" % spec.get("rank_verdict", "n/a"))
    return {"tstr_above_floor_pp": float(100 * gap)}
