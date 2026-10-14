"""Phase 7 ablation sweep -- the paper's main experiment.

7 ratios x 3 seeds = 21 runs. R0's 3 runs come from Phase 3; 18 remain.
Everything goes through the SAME code path, including R0, so nothing differs
but the data.

Non-negotiables enforced here:
  * no per-run hyperparameter tuning (train_classifier is reused unmodified)
  * no synthetic data in the validation or test sets
  * synthetic data regenerated fresh per seed
  * synthetic labels assigned by the Phase 2 physics labeller, not the GAN
  * CSV written after EVERY run
"""
import time
from datetime import datetime

import torch

from training.train_classifier import (
    train_classifier, append_result, _stratified_val_split, N_VAL, EPOCHS,
)
from training.train_gan import generate_synthetic
from utils.codebook import label_batch

# ratio -> synthetic count, with 500 real samples fixed in every run
RATIOS = [0, 25, 50, 75, 100, 150, 200]
SEEDS = [0, 1, 2]
CSV_PATH = "/kaggle/working/results/ablation_results.csv"


def run_sweep(seed_csi, seed_labels, test_csi, test_labels, codebook,
              G=None, seed_pos=None, test_norms=None,
              ratios=RATIOS, seeds=SEEDS, csv_path=CSV_PATH,
              device="cuda", verbose=True, epochs=None, n_val=None):
    """epochs/n_val exist ONLY for fast dry runs. Leave them at None for every
    real run -- the frozen recipe (100 epochs, 400/100 split) is what makes the
    ablation valid. Anything logged from a dry run must be deleted, not kept.
    """
    if epochs is None:
        epochs = EPOCHS
    if n_val is None:
        n_val = N_VAL
    n_real = seed_csi.shape[0]
    results = []

    for ratio in ratios:
        n_synth = int(round(n_real * ratio / 100.0))
        if n_synth > 0 and G is None:
            raise ValueError("ratio %d%% needs a trained generator" % ratio)

        for seed in seeds:
            t0 = time.time()

            # real-only validation split -- identical procedure in every run
            tr_idx, va_idx = _stratified_val_split(seed_labels, n_val, seed)
            train_csi = seed_csi[tr_idx]
            train_labels = seed_labels[tr_idx]
            val_csi, val_labels = seed_csi[va_idx], seed_labels[va_idx]

            if n_synth > 0:
                # fresh synthetic draw per seed
                synth_csi, _ = generate_synthetic(
                    G, seed_pos, n_synth, seed=seed, device=device
                )
                # labels come from the physics labeller, NOT the GAN
                synth_labels = label_batch(synth_csi, codebook)
                train_csi = torch.cat([train_csi, synth_csi])
                train_labels = torch.cat([train_labels, synth_labels])

            if verbose:
                print("\n=== R%d seed %d | train %d (%d real + %d synth) ===" % (
                    ratio, seed, len(train_labels), len(tr_idx), n_synth))

            m = train_classifier(
                train_csi, train_labels,
                val_csi, val_labels,
                test_csi, test_labels,
                seed=seed, codebook=codebook, test_norms=test_norms,
                device=device, epochs=epochs, verbose=verbose,
            )

            row = {
                "run_id": "R%d" % ratio, "synthetic_ratio": ratio, "seed": seed,
                "n_real": len(tr_idx), "n_synth": n_synth,
                "n_total": len(train_labels),
                "top1_acc": round(m["top1_acc"], 6),
                "top3_acc": round(m["top3_acc"], 6),
                "spectral_eff_pct": round(m["spectral_eff_pct"], 4),
                "best_val_acc": round(m["best_val_acc"], 6),
                "epochs": m["epochs"],
                "timestamp": datetime.now().isoformat(timespec="seconds"),
            }
            append_result(csv_path, row)      # checkpoint the CSV every run
            results.append(row)
            print("  -> top1 %.4f  top3 %.4f  SE%% %.2f  (%.0fs)" % (
                m["top1_acc"], m["top3_acc"], m["spectral_eff_pct"], time.time() - t0))

    return results


def aggregate(csv_path=CSV_PATH):
    """ratio -> mean +/- std per metric. Feeds Table III and Figure 5."""
    import pandas as pd
    df = pd.read_csv(csv_path)
    agg = df.groupby("synthetic_ratio").agg(
        top1_mean=("top1_acc", "mean"), top1_std=("top1_acc", "std"),
        top3_mean=("top3_acc", "mean"), top3_std=("top3_acc", "std"),
        se_mean=("spectral_eff_pct", "mean"), se_std=("spectral_eff_pct", "std"),
        n_runs=("seed", "count"),
    ).reset_index()
    return df, agg
