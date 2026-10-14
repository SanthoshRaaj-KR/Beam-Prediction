"""End-to-end smoke test on FABRICATED data -- no DeepMIMO required.

Run this from the repo root BEFORE the real data lands:
    python smoke_test.py

It exercises every module with tiny sizes so shape/API bugs surface in seconds
instead of two hours into a Kaggle session. It proves the plumbing works; it
says nothing about results.
"""
import os
import shutil
import tempfile

import numpy as np
import torch

from utils.codebook import (
    build_codebook, label_batch, majority_class_accuracy, N_ANT, N_SUB, N_BEAMS,
)
from models.generator import Generator
from models.critic import Critic, gradient_penalty
from training.train_classifier import train_classifier, _stratified_val_split
from training.train_gan import train_gan, generate_synthetic
from evaluation.run_ablation import run_sweep, aggregate
from evaluation import gan_quality_check as qc
from evaluation import make_figures as mf

DEV = "cuda" if torch.cuda.is_available() else "cpu"
TMP = tempfile.mkdtemp(prefix="smoke_")
PLOTS = os.path.join(TMP, "plots")


def fake_data(n, seed=0):
    """Structured fake CSI: a few random steering directions + noise, so the
    beam labels are non-degenerate and the classifier has something to learn."""
    rng = np.random.RandomState(seed)
    ang = rng.uniform(-1, 1, size=n)
    a = np.arange(N_ANT).reshape(1, -1)
    steer = np.exp(1j * np.pi * a * ang.reshape(-1, 1))            # [n, A]
    H = steer[:, :, None] * (1 + 0.1 * rng.randn(n, 1, N_SUB))
    H = H + 0.05 * (rng.randn(n, N_ANT, N_SUB) + 1j * rng.randn(n, N_ANT, N_SUB))
    csi = np.stack([H.real, H.imag], axis=1).astype(np.float32)    # [n,2,A,S]
    fro = np.linalg.norm(csi.reshape(n, -1), axis=1)
    csi = csi / fro.reshape(-1, 1, 1, 1)
    pos = np.stack([ang, rng.uniform(-1, 1, n)], axis=1).astype(np.float32)
    return (torch.from_numpy(csi), torch.from_numpy(pos),
            torch.from_numpy(fro.astype(np.float32)))


def main():
    print("device:", DEV, "| tmp:", TMP)
    cb = build_codebook()
    assert cb.shape == (N_BEAMS, N_ANT), cb.shape
    print("[1] codebook", tuple(cb.shape), cb.dtype)

    seed_csi, seed_pos, _ = fake_data(120, seed=0)
    test_csi, test_pos, test_norm = fake_data(80, seed=1)
    seed_labels = label_batch(seed_csi, cb)
    test_labels = label_batch(test_csi, cb)
    maj, maj_cls = majority_class_accuracy(seed_labels)
    populated = int((np.bincount(seed_labels.numpy(), minlength=N_BEAMS) > 0).sum())
    print("[2] labels: %d/32 populated | majority %.3f (class %d)" % (
        populated, maj, maj_cls))
    assert seed_labels.shape == (120,) and seed_labels.max() < N_BEAMS

    # labels must be invariant to per-sample Frobenius scaling
    scaled = seed_csi[:10] * 7.3
    assert torch.equal(label_batch(scaled, cb), seed_labels[:10]), \
        "labels changed under rescaling -- codebook math is wrong"
    print("[2b] label scale-invariance OK")

    # ---- Phase 4 shape verification (the mandatory gate) ----
    g, d = Generator().to(DEV), Critic().to(DEV)
    z = torch.randn(4, 100, device=DEV)
    c = (torch.rand(4, 2, device=DEV) * 2 - 1)
    fake = g(z, c)
    assert fake.shape == (4, 2, 64, 32), fake.shape
    assert d(fake, c).shape == (4, 1), d(fake, c).shape
    gp = gradient_penalty(d, seed_csi[:4].to(DEV), fake.detach(), c, torch.device(DEV))
    assert torch.isfinite(gp), "non-finite gradient penalty"
    print("[3] shapes OK | G %.2fM params | D %.2fM params | gp %.4f" % (
        sum(p.numel() for p in g.parameters()) / 1e6,
        sum(p.numel() for p in d.parameters()) / 1e6, gp.item()))

    # ---- Phase 5: tiny GAN run, then resume from checkpoint ----
    ck = os.path.join(TMP, "ckpt")
    G, D, w_hist = train_gan(seed_csi, seed_pos, n_iters=6, seed=42,
                             ckpt_dir=ck, device=DEV, log_every=5)
    assert len(w_hist) == 6
    G2, _, w2 = train_gan(seed_csi, seed_pos, n_iters=8, seed=42,
                          ckpt_dir=ck, device=DEV, log_every=5)
    assert len(w2) == 8, "resume did not continue the history: %d" % len(w2)
    print("[4] GAN train + resume OK (%d iters logged)" % len(w2))

    synth, scond = generate_synthetic(G2, seed_pos, 37, seed=0, device=DEV)
    assert synth.shape == (37, 2, 64, 32) and scond.shape == (37, 2)
    assert torch.isfinite(synth).all()
    print("[5] generate_synthetic", tuple(synth.shape))

    # ---- Phase 3: classifier ----
    tr, va = _stratified_val_split(seed_labels, 20, 0)
    m = train_classifier(
        seed_csi[tr], seed_labels[tr], seed_csi[va], seed_labels[va],
        test_csi, test_labels, seed=0, codebook=cb, test_norms=test_norm,
        device=DEV, epochs=3, verbose=False,
    )
    assert 0.0 <= m["top1_acc"] <= 1.0 and m["top1_acc"] <= m["top3_acc"]
    assert np.isfinite(m["spectral_eff_pct"])
    print("[6] classifier: top1 %.3f top3 %.3f SE%% %.2f (%.1fs)" % (
        m["top1_acc"], m["top3_acc"], m["spectral_eff_pct"], m["train_seconds"]))

    # ---- Phase 7: sweep across 3 ratios x 2 seeds ----
    csv = os.path.join(TMP, "results", "ablation_results.csv")
    rows = run_sweep(seed_csi, seed_labels, test_csi, test_labels, cb,
                     G=G2, seed_pos=seed_pos, test_norms=test_norm,
                     ratios=[0, 50, 100], seeds=[0, 1], csv_path=csv,
                     device=DEV, verbose=False, epochs=2, n_val=20)
    assert len(rows) == 6, len(rows)
    df, agg = aggregate(csv)
    assert len(df) == 6 and len(agg) == 3
    print("[7] sweep OK -> %d rows, %d ratios" % (len(df), len(agg)))

    # ---- Phase 6 + 8: quality checks and figures ----
    qc.test1_heatmaps(seed_csi, synth, n=4, plot_dir=PLOTS)
    qc.test2_power_profile(seed_csi, synth, plot_dir=PLOTS)
    t3 = qc.test3_discriminability(seed_csi, synth[:37], seeds=(0,),
                                   device=DEV, epochs=2, verbose=False)
    t4 = qc.test4_label_consistency(seed_labels, synth, cb, plot_dir=PLOTS)
    print("[8] quality: disc %.3f | label overlap %.3f" % (t3["mean"], t4["overlap"]))

    mf.fig_label_hist(seed_labels, maj, plot_dir=PLOTS)
    mf.fig_gan_training_curve(w2, plot_dir=PLOTS, table_dir=os.path.join(TMP, "results"))
    mf.fig_ablation_curve(agg, maj, plot_dir=PLOTS)
    mf.TABLE_DIR = os.path.join(TMP, "results")
    mf.table_main_results(agg, maj, table_dir=mf.TABLE_DIR)
    sig = mf.significance(df)
    assert np.isfinite(sig["improvement_pp"])
    print("[9] figures + table + stats OK")

    print("\nALL SMOKE TESTS PASSED -- plumbing is sound.")


if __name__ == "__main__":
    try:
        main()
    finally:
        shutil.rmtree(TMP, ignore_errors=True)
