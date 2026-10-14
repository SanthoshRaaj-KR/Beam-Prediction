"""Phase 1 -- DeepMIMO extraction & preprocessing.

!! THE ONE FUNCTION MARKED `TODO(PHASE0)` CANNOT BE WRITTEN BLIND. !!
Fill load_raw() from the Phase 0 help() output, then run. Everything else here
is complete.

Outputs to /kaggle/working/data/:
  seed_csi.pt  [500,2,64,32]   seed_pos.pt  [500,2]   seed_norm.pt  [500]
  test_csi.pt  [2000,2,64,32]  test_pos.pt  [2000,2]  test_norm.pt  [2000]
  seed_idx.npy, test_idx.npy
"""
import os

import numpy as np
import torch

OUT_DIR = "/kaggle/working/data"
SCENARIO = "o1_28"
SCENARIO_DIR = "/root/deepmimo_scenarios"
N_SEED, N_TEST = 500, 2000
N_ANT, N_SUB = 64, 32
SPLIT_SEED = 42


def load_raw():
    """TODO(PHASE0): fill from help(deepmimo.load) / help(deepmimo.ChannelParameters).

    Must return:
        H_all   -- complex array [n_users, N_ANT, N_SUB]
        pos_all -- float array   [n_users, 2]   (x, y; drop z, O1 is ground-level)

    Channel config to set via ChannelParameters:
        28 GHz (intrinsic to o1_28) | BS array 64-element ULA | UE 1 element
        32 OFDM subcarriers -- if unsupported, use the nearest value, DOCUMENT
        the real number, and propagate it everywhere (every 32 downstream).
    """
    raise NotImplementedError(
        "Fill load_raw() from the Phase 0 handoff: ChannelParameters fields, "
        "deepmimo.load/generate signature, and where the channel matrix and "
        "user positions live on the Dataset object."
    )


def preprocess(H_all, pos_all, out_dir=OUT_DIR):
    os.makedirs(out_dir, exist_ok=True)
    n_total = H_all.shape[0]

    # --- filter dead users: O1 contains blocked / out-of-coverage users whose
    # channels are all-zero or NaN. Report the count -- it goes in the paper.
    finite = np.isfinite(H_all).all(axis=(1, 2))
    nonzero = np.linalg.norm(H_all.reshape(n_total, -1), axis=1) > 0
    keep = finite & nonzero
    n_retained = int(keep.sum())
    n_dropped = n_total - n_retained
    print("users: %d total | %d dropped (dead) | %d retained" % (
        n_total, n_dropped, n_retained))
    assert n_retained >= N_SEED + N_TEST, "not enough valid users"

    H, pos = H_all[keep], pos_all[keep]

    # --- real/imag split -> [N,2,A,S], then per-sample Frobenius norm (stored)
    csi = np.stack([H.real, H.imag], axis=1).astype(np.float32)      # [N,2,A,S]
    fro = np.linalg.norm(csi.reshape(n_retained, -1), axis=1)        # [N]
    csi = csi / fro.reshape(-1, 1, 1, 1)

    # --- positions -> [-1,1] per axis, over the FULL retained set.
    # RECORD these bounds: GAN conditioning must use the identical transform.
    lo, hi = pos.min(axis=0), pos.max(axis=0)
    pos_n = (2 * (pos - lo) / (hi - lo) - 1).astype(np.float32)
    print("position bounds  x:[%.3f,%.3f]  y:[%.3f,%.3f]" % (lo[0], hi[0], lo[1], hi[1]))

    # --- disjoint split
    rng = np.random.RandomState(SPLIT_SEED)
    sel = rng.choice(n_retained, N_SEED + N_TEST, replace=False)
    seed_idx, test_idx = sel[:N_SEED], sel[N_SEED:]
    assert len(set(seed_idx) & set(test_idx)) == 0, "seed/test overlap!"

    for name, idx in (("seed", seed_idx), ("test", test_idx)):
        torch.save(torch.from_numpy(csi[idx]), "%s/%s_csi.pt" % (out_dir, name))
        torch.save(torch.from_numpy(pos_n[idx]), "%s/%s_pos.pt" % (out_dir, name))
        torch.save(torch.from_numpy(fro[idx].astype(np.float32)),
                   "%s/%s_norm.pt" % (out_dir, name))
        np.save("%s/%s_idx.npy" % (out_dir, name), idx)

    # --- checkpoint assertions
    sc = torch.load("%s/seed_csi.pt" % out_dir)
    tc = torch.load("%s/test_csi.pt" % out_dir)
    sp = torch.load("%s/seed_pos.pt" % out_dir)
    assert sc.shape == (N_SEED, 2, N_ANT, N_SUB), sc.shape
    assert tc.shape == (N_TEST, 2, N_ANT, N_SUB), tc.shape
    assert torch.isfinite(sc).all() and torch.isfinite(tc).all()
    assert sp.min() >= -1.0 and sp.max() <= 1.0
    print("Phase 1 OK:", tuple(sc.shape), tuple(tc.shape))

    return {"n_total": n_total, "n_dropped": n_dropped, "n_retained": n_retained,
            "pos_min": lo.tolist(), "pos_max": hi.tolist(),
            "n_ant": N_ANT, "n_sub": N_SUB}


if __name__ == "__main__":
    H_all, pos_all = load_raw()
    print(preprocess(H_all, pos_all))
