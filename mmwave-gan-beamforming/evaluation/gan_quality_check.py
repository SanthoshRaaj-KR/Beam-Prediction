"""Phase 6 -- GAN quality check.

A negative result here is publishable; a hidden negative result is misconduct.
Report what you find.

ONE-DAY CUT: Test 2 is trimmed to the antenna-power-profile comparison only.
Tests 3 and 4 are the ones that carry Section V, so they are kept in full.
"""
import os

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

from models.beam_classifier import BeamCNN
from training.train_classifier import set_all_seeds
from utils.codebook import csi_to_complex, label_batch, N_BEAMS

PLOT_DIR = "/kaggle/working/results/plots"


def _mag(csi):
    """|H| magnitude maps [N,A,S] from a [N,2,A,S] real tensor."""
    return torch.sqrt(csi[:, 0] ** 2 + csi[:, 1] ** 2)


def test1_heatmaps(real_csi, synth_csi, n=8, plot_dir=PLOT_DIR):
    """Figure 3 -- real (top row) vs synthetic (bottom row), shared color scale."""
    os.makedirs(plot_dir, exist_ok=True)
    r, s = _mag(real_csi[:n]).numpy(), _mag(synth_csi[:n]).numpy()
    vmax = float(max(r.max(), s.max()))

    fig, axes = plt.subplots(2, n, figsize=(2.0 * n, 4.6))
    for i in range(n):
        for row, (data, lab) in enumerate(((r, "Real"), (s, "Synthetic"))):
            ax = axes[row, i]
            ax.imshow(data[i], aspect="auto", vmin=0, vmax=vmax, cmap="viridis")
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.set_ylabel(lab, fontsize=11)
    fig.suptitle("CSI magnitude |H| (64 antennas x 32 subcarriers)", fontsize=12)
    fig.tight_layout()
    path = os.path.join(plot_dir, "fig_real_vs_synth.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("Test 1 ->", path)
    return path


def test2_power_profile(real_csi, synth_csi, plot_dir=PLOT_DIR):
    """Antenna-domain power profile, real vs synthetic (mean +/- std).

    Post-normalization ||H||_F is ~1 by construction, so compare a STRUCTURAL
    statistic instead: mean over subcarriers of |H[a,:]|^2, a 64-vector.
    """
    os.makedirs(plot_dir, exist_ok=True)

    def profile(csi):
        p = (_mag(csi) ** 2).mean(dim=2).numpy()      # [N, A]
        return p.mean(axis=0), p.std(axis=0), p

    rm, rs, rp = profile(real_csi)
    sm, ss, sp = profile(synth_csi)

    a = np.arange(len(rm))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(a, rm, label="Real", color="#1f77b4")
    ax.fill_between(a, rm - rs, rm + rs, alpha=0.25, color="#1f77b4")
    ax.plot(a, sm, label="Synthetic", color="#d62728")
    ax.fill_between(a, sm - ss, sm + ss, alpha=0.25, color="#d62728")
    ax.set_xlabel("Antenna index"); ax.set_ylabel("Mean power")
    ax.set_title("Antenna-domain power profile (mean $\\pm$ 1 std)")
    ax.legend(); fig.tight_layout()
    path = os.path.join(plot_dir, "fig_power_profile.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    # scalar summary: antenna-domain power spread, compared by Wasserstein distance
    try:
        from scipy.stats import wasserstein_distance
        w = float(wasserstein_distance(rp.std(axis=1), sp.std(axis=1)))
    except Exception:
        w = float("nan")
    print("Test 2 -> %s | Wasserstein(power spread) = %.5f" % (path, w))
    return {"plot": path, "wasserstein_power_spread": w}


def test3_discriminability(real_csi, synth_csi, seeds=(0, 1, 2), device="cuda",
                           epochs=30, verbose=True):
    """THE headline quantitative number.

    Balanced 500 real + 500 synthetic, 80/20 split, 3 seeds, mean +/- std.
      ~50-60%  near-indistinguishable, strong realism
      60-80%   partially realistic; note as a limitation
      >90%     easily distinguishable; report honestly -- the ablation is still
               valid and this becomes a central discussion point
    """
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    n = min(len(real_csi), len(synth_csi))
    X = torch.cat([real_csi[:n], synth_csi[:n]])
    y = torch.cat([torch.zeros(n, dtype=torch.long), torch.ones(n, dtype=torch.long)])

    accs = []
    for seed in seeds:
        set_all_seeds(seed)
        perm = torch.randperm(len(X))
        Xs, ys = X[perm], y[perm]
        cut = int(0.8 * len(Xs))
        Xtr, ytr, Xte, yte = Xs[:cut], ys[:cut], Xs[cut:], ys[cut:]

        model = BeamCNN(n_beams=2).to(dev)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        crit = nn.CrossEntropyLoss()
        loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=64, shuffle=True)

        model.train()
        for _ in range(epochs):
            for xb, yb in loader:
                xb, yb = xb.to(dev), yb.to(dev)
                opt.zero_grad(); crit(model(xb), yb).backward(); opt.step()

        model.eval()
        with torch.no_grad():
            pred = model(Xte.to(dev)).argmax(1).cpu()
        acc = (pred == yte).float().mean().item()
        accs.append(acc)
        if verbose:
            print("  seed %d: discriminator acc %.4f" % (seed, acc))

    mean, std = float(np.mean(accs)), float(np.std(accs))
    if mean <= 0.60:
        verdict = "near-indistinguishable (strong realism)"
    elif mean <= 0.80:
        verdict = "partially realistic -- note as a limitation"
    elif mean <= 0.90:
        verdict = "largely distinguishable -- discuss openly"
    else:
        verdict = "easily distinguishable -- report honestly; expect weak augmentation gains"
    print("Test 3 -> GAN-test acc %.4f +/- %.4f  [%s]" % (mean, std, verdict))
    return {"accs": accs, "mean": mean, "std": std, "verdict": verdict}


def test4_label_consistency(real_labels, synth_csi, codebook, plot_dir=PLOT_DIR):
    """Figure 4 -- does the GAN reproduce the real optimal-beam distribution?

    Direct evidence of task-relevant structure rather than matched low-order
    statistics. Reports histogram-overlap coefficient in [0,1].
    """
    os.makedirs(plot_dir, exist_ok=True)
    synth_labels = label_batch(synth_csi, codebook)

    rl = real_labels.numpy() if torch.is_tensor(real_labels) else np.asarray(real_labels)
    sl = synth_labels.numpy()
    rh = np.bincount(rl, minlength=N_BEAMS) / len(rl)
    sh = np.bincount(sl, minlength=N_BEAMS) / len(sl)
    overlap = float(np.minimum(rh, sh).sum())

    x = np.arange(N_BEAMS)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - 0.2, rh, width=0.4, label="Real", color="#1f77b4")
    ax.bar(x + 0.2, sh, width=0.4, label="Synthetic", color="#d62728")
    ax.set_xlabel("Beam index"); ax.set_ylabel("Fraction of samples")
    ax.set_title("Optimal-beam distribution (overlap = %.3f)" % overlap)
    ax.legend(); fig.tight_layout()
    path = os.path.join(plot_dir, "fig_label_dist.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print("Test 4 -> %s | label-distribution overlap = %.4f" % (path, overlap))
    return {"plot": path, "overlap": overlap,
            "synth_labels": synth_labels,
            "n_populated_synth": int((sh > 0).sum())}
