"""Phase 1 -- DeepMIMO extraction & preprocessing.

Written against DeepMIMO 4.0.5 / scenario o1_28. Key API facts (see the
execution guide §0.5):

  * Use generate(), not load(). load() with no restriction pulls all 54 TX-RX
    pairs (~1.18M users per TX set); we need 2500.
  * trim_params['idxs'] subsets UEs BEFORE channels are computed. We pass
    explicit indices rather than idxs_mode='uniform' so the selection is
    deterministic and needs no guesswork about idxs_kwargs.
  * ofdm['subcarriers'] is the FFT size (512). ofdm['selected_subcarriers']
    chooses which ones are returned -- that is how we land on exactly 32.
  * bs_antenna['shape'] DEFAULTS TO [8,1]. It must be set to [64,1] explicitly
    or you silently train on an 8-element array.

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

# --- scenario layout (from deepmimo.summary('o1_28')) ---
# RX sets: 0 (RX_1, 497931 users) | 1 (RX_2, 199281) | 2 (RX_3, 487711)
# TX sets: 3..20, one point each. Set 4 is TX_3 -- BS3 is the commonly used
# transmitter in O1 beam-prediction work. Any single BS is defensible; just
# document the choice.
TX_SET, RX_SET = 4, 0
RX_SET_SIZE = 497931
N_SUBSAMPLE = 6000      # margin over the 2500 needed, for dead-user dropout
N_PATHS = 25
FFT_SIZE = 512


def build_ch_params(deepmimo):
    """ChannelParameters for a 64-element ULA BS, single-antenna UE, 32 subcarriers."""
    p = deepmimo.ChannelParameters()
    p.bs_antenna.shape = np.array([N_ANT, 1])     # 64-element ULA (default is [8,1]!)
    p.ue_antenna.shape = np.array([1, 1])         # single-antenna RX -> H stays [64,S]
    p.ofdm.subcarriers = FFT_SIZE
    # Spread the 32 selected subcarriers across the full 512-bin / 10 MHz band.
    # Taking 0..31 contiguously would sample a narrow, highly correlated slice,
    # leaving the subcarrier axis nearly flat and giving the GAN little to learn
    # along it. Document this choice in the paper.
    p.ofdm.selected_subcarriers = np.linspace(0, FFT_SIZE - 1, N_SUB).astype(int)
    p.freq_domain = 1
    p.num_paths = N_PATHS
    return p


def _attr(obj, name, default=None):
    """getattr() that survives DeepMIMO's Dataset.

    Dataset.__getattr__ raises KeyError for an unknown name, not the
    AttributeError the language expects. hasattr() and getattr(..., default)
    only swallow AttributeError, so both PROPAGATE that KeyError and crash the
    caller. Never probe a DeepMIMO Dataset with hasattr/getattr -- use this.
    """
    try:
        return getattr(obj, name)
    except Exception:
        return default


def _keys(obj):
    """Best-effort list of what a Dataset actually holds, for error messages."""
    for probe in ("keys", "_data"):
        got = _attr(obj, probe)
        if callable(got):
            try:
                return sorted(got())
            except Exception:
                pass
        elif isinstance(got, dict):
            return sorted(got.keys())
    return [k for k in dir(obj) if not k.startswith("_")]


def _as_dataset(obj):
    """generate() may return a Dataset or a 1-element MacroDataset."""
    inner = _attr(obj, "datasets")
    if inner is None:
        return obj                      # already a plain Dataset
    return inner[0] if len(inner) else obj


def _first_attr(obj, names, what):
    """Fetch the first attribute that exists, or fail with the real key list."""
    for n in names:
        got = _attr(obj, n)
        if got is not None:
            return got, n
    raise KeyError(
        "could not find %s on the dataset under any of %s. Available keys: %s"
        % (what, names, _keys(obj)))


def _normalize_channel_shape(ch):
    """Squeeze a DeepMIMO channel array down to [n_users, N_ANT, N_SUB] complex.

    The middle axes carry the RX antenna (1) and TX antenna (64) dimensions;
    their exact order is not relied upon -- singleton axes are dropped and the
    remaining two are oriented by matching N_ANT and N_SUB.
    """
    ch = np.asarray(ch)
    print("raw channel shape:", ch.shape, ch.dtype)
    n_users = ch.shape[0]

    # drop every singleton axis after the user axis
    rest = [s for s in ch.shape[1:] if s != 1]
    ch = ch.reshape((n_users,) + tuple(rest))

    if ch.ndim != 3:
        raise ValueError(
            "expected 3 dims after squeezing, got %s. Inspect the raw shape "
            "above and adjust _normalize_channel_shape()." % (ch.shape,))

    if ch.shape[1] == N_SUB and ch.shape[2] == N_ANT:
        ch = ch.transpose(0, 2, 1)                 # [N,S,A] -> [N,A,S]
    elif not (ch.shape[1] == N_ANT and ch.shape[2] == N_SUB):
        raise ValueError(
            "cannot orient channel to [N,%d,%d]; got %s" % (N_ANT, N_SUB, ch.shape))

    print("oriented channel shape:", ch.shape)
    return ch


def load_raw(n_subsample=N_SUBSAMPLE):
    """Generate channels for a deterministic UE subset of one BS / one RX set.

    Returns:
        H_all   -- complex array [n_users, N_ANT, N_SUB]
        pos_all -- float array   [n_users, 2]   (x, y; z dropped, O1 is ground-level)
    """
    import deepmimo

    # Deterministic spatially-spread subset, chosen BEFORE channel computation
    # so we never pay for the ~498k users we do not use.
    idxs = np.linspace(0, RX_SET_SIZE - 1, n_subsample).astype(int)

    ch_params = build_ch_params(deepmimo)
    print("generating %d UEs | BS set %d | RX set %d | %d subcarriers of %d"
          % (n_subsample, TX_SET, RX_SET, N_SUB, FFT_SIZE))

    out = deepmimo.generate(
        SCENARIO,
        load_params={"tx_sets": {TX_SET: "all"},
                     "rx_sets": {RX_SET: "all"},
                     "max_paths": N_PATHS},
        trim_params={"idxs": idxs},
        ch_params=ch_params,
    )
    ds = _as_dataset(out)
    print("dataset:", type(ds).__name__, "| n_ue:", _attr(ds, "n_ue", "?"))

    ch, ch_key = _first_attr(ds, ("ch", "channel", "channels"), "the channel matrix")
    pos, pos_key = _first_attr(ds, ("rx_pos", "rx_loc", "pos"), "the RX positions")
    print("using keys: channel=%r positions=%r" % (ch_key, pos_key))

    H_all = _normalize_channel_shape(ch)
    pos_all = np.asarray(pos)[:, :2].astype(np.float64)   # drop z
    print("positions:", pos_all.shape)

    assert H_all.shape[0] == pos_all.shape[0], (H_all.shape, pos_all.shape)
    return H_all, pos_all


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
