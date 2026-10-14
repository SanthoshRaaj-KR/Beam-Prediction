"""WGAN-GP training (Phase 5), with checkpoint + resume.

Frozen hyperparameters -- these are the published WGAN-GP values. Do NOT
substitute DCGAN defaults; beta2=0.9 matters, 0.999 is specifically wrong here.

ONE-DAY CUT: N_ITERS defaults to 3000 instead of 5000. Checkpoints land every
500 iters, so if you are short on time you can stop early and still have a
usable generator.
"""
import os
import glob

import numpy as np
import torch

from models.generator import Generator
from models.critic import Critic, gradient_penalty
from training.train_classifier import set_all_seeds

# ---- FROZEN ----
LAMBDA_GP = 10.0
N_CRITIC = 5
LR = 1e-4
BETAS = (0.5, 0.9)      # NOT (0.5, 0.999)
BATCH_SIZE = 64
Z_DIM = 100
N_ITERS = 3000
CKPT_EVERY = 500
CKPT_DIR = "/kaggle/working/checkpoints"


def _latest_ckpt(ckpt_dir):
    files = glob.glob(os.path.join(ckpt_dir, "gan_*.pt"))
    if not files:
        return None
    return max(files, key=lambda p: int(os.path.basename(p).split("_")[1].split(".")[0]))


def train_gan(seed_csi, seed_pos, n_iters=N_ITERS, seed=42,
              ckpt_dir=CKPT_DIR, device="cuda", log_every=100):
    """Train the conditional WGAN-GP on the 500-sample real seed set.

    Conditioning: positions are drawn from the real seed set WITH replacement,
    which keeps the conditioning distribution identical to the real data.
    Document this choice in the paper.
    """
    set_all_seeds(seed)
    os.makedirs(ckpt_dir, exist_ok=True)
    dev = torch.device(device if torch.cuda.is_available() else "cpu")

    G, D = Generator(z_dim=Z_DIM).to(dev), Critic().to(dev)
    optG = torch.optim.Adam(G.parameters(), lr=LR, betas=BETAS)
    optD = torch.optim.Adam(D.parameters(), lr=LR, betas=BETAS)

    real_all = seed_csi.to(dev)
    pos_all = seed_pos.to(dev)
    n = real_all.shape[0]

    start_it, w_hist = 0, []
    ck = _latest_ckpt(ckpt_dir)
    if ck:
        st = torch.load(ck, map_location=dev)
        G.load_state_dict(st["G"]); D.load_state_dict(st["D"])
        optG.load_state_dict(st["optG"]); optD.load_state_dict(st["optD"])
        start_it, w_hist = st["it"] + 1, st.get("w_hist", [])
        print(f"resumed from {ck} at iter {start_it}")

    def sample_real(bs):
        i = torch.randint(0, n, (bs,), device=dev)
        return real_all[i], pos_all[i]

    for it in range(start_it, n_iters):
        for _ in range(N_CRITIC):
            real, cond = sample_real(BATCH_SIZE)
            z = torch.randn(BATCH_SIZE, Z_DIM, device=dev)
            fake = G(z, cond).detach()
            gp = gradient_penalty(D, real, fake, cond, dev)
            d_real, d_fake = D(real, cond).mean(), D(fake, cond).mean()
            d_loss = d_fake - d_real + LAMBDA_GP * gp
            optD.zero_grad(); d_loss.backward(); optD.step()

        real, cond = sample_real(BATCH_SIZE)
        z = torch.randn(BATCH_SIZE, Z_DIM, device=dev)
        g_loss = -D(G(z, cond), cond).mean()
        optG.zero_grad(); g_loss.backward(); optG.step()

        # Wasserstein estimate -- the only meaningful progress signal.
        # Raw d_loss/g_loss values are NOT progress indicators in WGAN.
        w_est = (d_real - d_fake).item()
        w_hist.append({"it": it, "w_est": w_est,
                       "d_loss": d_loss.item(), "g_loss": g_loss.item()})

        if not np.isfinite(w_est):
            raise RuntimeError(
                f"NaN at iter {it}. This is a bug, not bad luck. Check: "
                "gradient penalty create_graph=True, no BatchNorm in critic, LR not raised."
            )

        if it % log_every == 0:
            print(f"it {it:5d}  w_est {w_est:+.4f}  d_loss {d_loss.item():+.3f}  g_loss {g_loss.item():+.3f}")

        if (it + 1) % CKPT_EVERY == 0 or it == n_iters - 1:
            torch.save({"it": it, "G": G.state_dict(), "D": D.state_dict(),
                        "optG": optG.state_dict(), "optD": optD.state_dict(),
                        "w_hist": w_hist},
                       os.path.join(ckpt_dir, f"gan_{it}.pt"))
            print(f"  checkpoint -> gan_{it}.pt")

    return G, D, w_hist


@torch.no_grad()
def generate_synthetic(G, seed_pos, n_samples, seed, z_dim=Z_DIM, device="cuda", batch=256):
    """Generate n_samples synthetic CSI, conditioned on real seed positions
    drawn with replacement (same procedure as training).

    Regenerate fresh per seed -- that is what makes the Phase 7 error bars
    meaningful (seed varies both weight init and the synthetic draw).
    """
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    G = G.to(dev).eval()
    g = torch.Generator(device="cpu").manual_seed(seed)

    out, conds = [], []
    done = 0
    while done < n_samples:
        bs = min(batch, n_samples - done)
        idx = torch.randint(0, seed_pos.shape[0], (bs,), generator=g)
        c = seed_pos[idx].to(dev)
        z = torch.randn(bs, z_dim, generator=g).to(dev)
        out.append(G(z, c).cpu())
        conds.append(c.cpu())
        done += bs
    return torch.cat(out), torch.cat(conds)
