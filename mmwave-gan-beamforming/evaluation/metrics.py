"""Evaluation metrics (Phases 3 & 8).

SNR is FIXED at 10 dB -- a reported assumption, not a tuned parameter. State it
in the paper.
"""
import numpy as np
import torch

SNR_DB = 10.0
SNR_LIN = 10 ** (SNR_DB / 10.0)     # 10.0


@torch.no_grad()
def top_k_accuracy(logits, labels, k=1):
    topk = logits.topk(k, dim=1).indices                 # [B,k]
    hit = (topk == labels.view(-1, 1)).any(dim=1)
    return hit.float().mean().item()


@torch.no_grad()
def spectral_efficiency_pct(logits, csi, labels, codebook, norms=None, snr_lin=SNR_LIN):
    """mean(SE_pred / SE_opt) * 100 over the test set.

    csi:      [N,2,A,S] normalized real tensor
    norms:    [N] stored Frobenius norms -- restores true scale before gains.
              Pass None only if you deliberately report normalized-scale SE.
    codebook: [n_beams, A] complex
    """
    from utils.codebook import csi_to_complex

    preds = logits.argmax(1).cpu().numpy()
    H = csi_to_complex(csi)                                      # [N,A,S] complex
    if norms is not None:
        n = norms.detach().cpu().numpy() if torch.is_tensor(norms) else np.asarray(norms)
        H = H * n.reshape(-1, 1, 1)                              # restore true scale

    W = codebook.detach().cpu().numpy() if torch.is_tensor(codebook) else codebook
    y = labels.detach().cpu().numpy() if torch.is_tensor(labels) else np.asarray(labels)

    ratios = []
    for i in range(H.shape[0]):
        gains = np.abs(W.conj() @ H[i]) ** 2                     # [n_beams, S]
        g = gains.mean(axis=1)
        se_pred = np.log2(1.0 + snr_lin * g[preds[i]])
        se_opt = np.log2(1.0 + snr_lin * g[y[i]])
        if se_opt > 0:
            ratios.append(se_pred / se_opt)
    return float(np.mean(ratios) * 100.0)
