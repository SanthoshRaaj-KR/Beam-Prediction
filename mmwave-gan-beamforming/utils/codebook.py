"""DFT beam codebook and physics-based optimal-beam labelling (Phase 2).

Frozen per IMPLEMENTATION_PHASES.md Phase 2. The labeller here is the ONLY
source of beam labels in the project -- it labels real samples AND synthetic
samples (the GAN does not produce labels; they are derived analytically from
the generated channel).
"""
import numpy as np
import torch

N_ANT = 64      # BS ULA elements
N_BEAMS = 32    # codebook size == number of classes
N_SUB = 32      # OFDM subcarriers (update here if Phase 1 used another value)


def build_codebook(n_ant=N_ANT, n_beams=N_BEAMS):
    """Return the DFT beam codebook as [n_beams, n_ant] complex.

    Row k is beam k:  w_k[n] = exp(j*2*pi*n*k/n_beams) / sqrt(n_ant)
    """
    n = np.arange(n_ant).reshape(-1, 1)        # [n_ant, 1]
    k = np.arange(n_beams).reshape(1, -1)      # [1, n_beams]
    W = np.exp(1j * 2 * np.pi * n * k / n_beams) / np.sqrt(n_ant)   # [n_ant, n_beams]
    return torch.from_numpy(W.T.copy())        # [n_beams, n_ant], row k = beam k


def csi_to_complex(csi):
    """[*, 2, A, S] real tensor -> [*, A, S] complex numpy array."""
    if torch.is_tensor(csi):
        csi = csi.detach().cpu().numpy()
    return csi[..., 0, :, :] + 1j * csi[..., 1, :, :]


def beam_gains(H_complex, codebook):
    """Per-beam average beamforming gain.

    H_complex: [A, S] complex          codebook: [n_beams, A] complex
    returns:   [n_beams] float   -- mean_s |w_k^H h_s|^2
    """
    if torch.is_tensor(codebook):
        codebook = codebook.detach().cpu().numpy()
    gains = np.abs(codebook.conj() @ H_complex) ** 2    # [n_beams, S]
    return gains.mean(axis=1)


def optimal_beam(H_complex, codebook):
    """Ground-truth label: argmax over beams of the subcarrier-averaged gain."""
    return int(beam_gains(H_complex, codebook).argmax())


def label_batch(csi, codebook):
    """Label a whole [N, 2, A, S] tensor -> LongTensor [N].

    Per-sample Frobenius scaling does not change the argmax, so this is safe to
    run on normalized CSI (Phase 2 verifies this on 10 samples).
    """
    H = csi_to_complex(csi)                              # [N, A, S]
    labels = [optimal_beam(H[i], codebook) for i in range(H.shape[0])]
    return torch.tensor(labels, dtype=torch.long)


def majority_class_accuracy(labels):
    """Trivial baseline every ablation run must beat. Returns (acc, class_id)."""
    if torch.is_tensor(labels):
        labels = labels.detach().cpu().numpy()
    counts = np.bincount(labels, minlength=N_BEAMS)
    return float(counts.max()) / len(labels), int(counts.argmax())
