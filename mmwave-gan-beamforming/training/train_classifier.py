"""Frozen classifier training recipe (Phase 3).

CRITICAL: this function is written once and reused UNMODIFIED by Phase 7 for all
21 runs. Only the training data changes. Never tune hyperparameters per run --
that is what makes the ablation valid.

Two hard rules enforced here:
  * the validation set is drawn from REAL data only, in every run
  * the test set is touched exactly ONCE per run, at the very end
"""
import csv
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from models.beam_classifier import BeamCNN
from evaluation.metrics import top_k_accuracy, spectral_efficiency_pct

# ---- FROZEN RECIPE (do not edit after Phase 3) ----
EPOCHS = 100
BATCH_SIZE = 64
LR = 1e-3
N_VAL = 100          # 400/100 split within the 500 real samples
CSV_COLUMNS = [
    "run_id", "synthetic_ratio", "seed", "n_real", "n_synth", "n_total",
    "top1_acc", "top3_acc", "spectral_eff_pct", "best_val_acc", "epochs", "timestamp",
]


def set_all_seeds(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)


def _stratified_val_split(labels, n_val, seed):
    """Stratify if every class allows it, else fall back to a plain shuffle.

    With 32 classes over 500 samples some classes have a single member, which
    makes strict stratification impossible -- the fallback is expected.
    """
    n = len(labels)
    try:
        from sklearn.model_selection import train_test_split
        idx = np.arange(n)
        tr, va = train_test_split(
            idx, test_size=n_val, random_state=seed, stratify=labels.numpy()
        )
        return tr, va
    except Exception:
        rng = np.random.RandomState(seed)
        idx = rng.permutation(n)
        return idx[n_val:], idx[:n_val]


def train_classifier(
    train_csi, train_labels,
    val_csi, val_labels,
    test_csi, test_labels,
    seed,
    codebook=None, test_norms=None,
    device="cuda", epochs=EPOCHS, verbose=True,
):
    """Train BeamCNN on (train_csi, train_labels); return a metrics dict.

    val_* MUST be real-only. test_* is evaluated once, after model selection.
    """
    set_all_seeds(seed)
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    model = BeamCNN(n_beams=32).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    loader = DataLoader(
        TensorDataset(train_csi, train_labels),
        batch_size=BATCH_SIZE, shuffle=True, drop_last=False,
    )
    val_csi_d, val_labels_d = val_csi.to(device), val_labels.to(device)

    best_val, best_state = -1.0, None
    t0 = time.time()

    for ep in range(epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_acc = top_k_accuracy(model(val_csi_d), val_labels_d, k=1)
        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

        if verbose and (ep + 1) % 20 == 0:
            print(f"  ep {ep+1:3d}/{epochs}  loss {loss.item():.3f}  val@1 {val_acc:.3f}")

    # ---- model selection done; the test set is touched exactly once ----
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits = model(test_csi.to(device)).cpu()
    test_labels_c = test_labels.cpu()

    out = {
        "top1_acc": top_k_accuracy(logits, test_labels_c, k=1),
        "top3_acc": top_k_accuracy(logits, test_labels_c, k=3),
        "best_val_acc": best_val,
        "epochs": epochs,
        "train_seconds": time.time() - t0,
        "spectral_eff_pct": float("nan"),
    }
    if codebook is not None:
        out["spectral_eff_pct"] = spectral_efficiency_pct(
            logits, test_csi, test_labels_c, codebook, norms=test_norms
        )
    return out


def append_result(csv_path, row):
    """Append one run to the ablation CSV. Called after EVERY run, not at the end."""
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    new = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in CSV_COLUMNS})
