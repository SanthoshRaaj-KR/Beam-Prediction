"""One-command driver for the whole study.

    python run_all.py              # run everything that is not already done
    python run_all.py --only gan   # run a single stage
    python run_all.py --status     # print what is done and what the numbers are

Why a script instead of notebook cells: a subprocess gets a fresh interpreter
every time, so edits to the source are always picked up (no stale modules in
`sys.modules`), and no kernel restart is needed after `pip install deepmimo`.

EVERY STAGE IS RESUMABLE. Each one checks for its own output on disk and skips
itself if it is already there, so after a Kaggle session dies you re-run the
same command and it continues where it stopped. The GAN resumes mid-training
from its newest checkpoint; the ablation sweep skips (ratio, seed) pairs that
are already rows in the results CSV.

Stage order:
    scenario  make deepmimo's o1_28 importable (symlink, or download if absent)
    data      Phase 1 -- generate channels, filter dead users, split 500/2000
    labels    Phase 2 -- DFT codebook, physics beam labels, Figure 1
    baseline  Phase 3 -- R0 (real-only) x 3 seeds. HARD GATE on majority class.
    gan       Phase 5 -- WGAN-GP, 3000 iters, checkpointed every 500
    quality   Phase 6 -- 4 GAN realism checks (test3 is the headline number)
    sweep     Phase 7 -- the remaining 18 runs (6 ratios x 3 seeds)
    figures   Phase 8 -- aggregate, Figures 2/5, Table III, significance tests

Recorded numbers accumulate in <work>/state.json, and `--status` prints them
grouped by the paper placeholders they fill.
"""
import argparse
import json
import os
import sys
import time
import traceback

# Make the repo importable no matter where this script is invoked from.
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

SCENARIO = "o1_28"
STAGES = ["scenario", "data", "labels", "baseline", "gan",
          "quality", "mechanism", "sweep", "figures"]

# "Clearly beats" for the Phase 3 gate, in absolute accuracy. A working
# classifier on 32 beams should clear the majority class by far more than this;
# the margin exists to catch a model that has only learned the class prior.
GATE_MARGIN = 0.05


# --------------------------------------------------------------------------
# state
# --------------------------------------------------------------------------

class State:
    """Durable record of finished stages and the numbers the paper needs."""

    def __init__(self, path):
        self.path = path
        self.d = {"done": [], "numbers": {}}
        if os.path.isfile(path):
            with open(path) as f:
                self.d = json.load(f)

    def save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(self.d, f, indent=2, default=str)

    def is_done(self, stage):
        return stage in self.d["done"]

    def finish(self, stage):
        if stage not in self.d["done"]:
            self.d["done"].append(stage)
        self.save()

    def record(self, **kw):
        self.d["numbers"].update(kw)
        self.save()

    def num(self, key, default=None):
        return self.d["numbers"].get(key, default)


# --------------------------------------------------------------------------
# lazily-loaded artifacts
# --------------------------------------------------------------------------

class Ctx:
    """Loads each artifact from disk on first use.

    Every stage reads what it needs through this object rather than receiving
    it from the stage before, which is what lets any stage run standalone after
    a session restart.
    """

    def __init__(self, args, state):
        self.args = args
        self.state = state
        self.work = args.work_dir
        self.data_dir = os.path.join(self.work, "data")
        self.ckpt_dir = os.path.join(self.work, "checkpoints")
        self.results_dir = os.path.join(self.work, "results")
        self.plot_dir = os.path.join(self.results_dir, "plots")
        self.csv_path = os.path.join(self.results_dir, "ablation_results.csv")
        self._cache = {}

    # -- tensors from Phase 1/2 ------------------------------------------
    def _load(self, name):
        import torch
        if name not in self._cache:
            path = os.path.join(self.data_dir, name + ".pt")
            if not os.path.isfile(path):
                raise SystemExit(
                    "missing %s -- run the earlier stage first "
                    "(python run_all.py --only data)" % path)
            self._cache[name] = torch.load(path)
        return self._cache[name]

    seed_csi = property(lambda s: s._load("seed_csi"))
    test_csi = property(lambda s: s._load("test_csi"))
    seed_pos = property(lambda s: s._load("seed_pos"))
    test_pos = property(lambda s: s._load("test_pos"))
    seed_norm = property(lambda s: s._load("seed_norm"))
    test_norm = property(lambda s: s._load("test_norm"))
    seed_labels = property(lambda s: s._load("seed_labels"))
    test_labels = property(lambda s: s._load("test_labels"))

    @property
    def codebook(self):
        if "codebook" not in self._cache:
            from utils.codebook import build_codebook
            self._cache["codebook"] = build_codebook()
        return self._cache["codebook"]

    @property
    def majority(self):
        """Majority-class accuracy over the full labelled set."""
        import torch
        from utils.codebook import majority_class_accuracy
        if "majority" not in self._cache:
            allk = torch.cat([self.seed_labels, self.test_labels])
            self._cache["majority"] = majority_class_accuracy(allk)
        return self._cache["majority"]

    # -- GAN -------------------------------------------------------------
    @property
    def _gan_ckpt(self):
        from training.train_gan import _latest_ckpt
        ck = _latest_ckpt(self.ckpt_dir)
        if ck is None:
            raise SystemExit(
                "no GAN checkpoint in %s -- run: python run_all.py --only gan"
                % self.ckpt_dir)
        return ck

    @property
    def generator(self):
        """Rebuild G from the newest checkpoint (works in a fresh process)."""
        import torch
        from models.generator import Generator
        if "G" not in self._cache:
            ck = self._gan_ckpt
            st = torch.load(ck, map_location="cpu")
            G = Generator(z_dim=100)
            G.load_state_dict(st["G"])
            G.eval()
            print("loaded generator from %s (iter %d)" % (ck, st["it"]))
            self._cache["G"] = G
        return self._cache["G"]

    @property
    def w_hist(self):
        import torch
        st = torch.load(self._gan_ckpt, map_location="cpu")
        return st.get("w_hist", [])


# --------------------------------------------------------------------------
# stages
# --------------------------------------------------------------------------

def _configure_deepmimo(args):
    """Point deepmimo at an ABSOLUTE scenarios dir. Must run in every process.

    `deepmimo.config` is a singleton whose `set()` only writes an in-memory
    dict -- nothing is persisted. So each new interpreter starts again with the
    default `scenarios_folder == "deepmimo_scenarios"`, a RELATIVE path
    resolved against the current working directory. Two consequences:

      * `--only data` in a fresh process would look for the scenario under
        whatever cwd it happened to inherit, and report it missing.
      * The 21 GB scenario must NOT live under /kaggle/working (20 GB cap), so
        the directory has to be chosen explicitly rather than inherited.

    Setting it absolute, on every run, is what makes both problems go away.
    """
    import deepmimo
    deepmimo.config("scenarios_folder", args.scenario_dir)
    return deepmimo


def _is_scenario_dir(path, min_files=100):
    """A real extracted scenario (o1_28 has 4161 files), or a nested copy.

    Returns the directory that actually holds the files, or None. The nested
    case shows up when a previous download's final `shutil.move()` landed
    inside an existing target, giving `<dir>/o1_28/o1_28`.
    """
    if not os.path.isdir(path):
        return None
    try:
        entries = os.listdir(path)
    except OSError:
        return None
    if len(entries) >= min_files:
        return path
    nested = os.path.join(path, SCENARIO)
    if os.path.isdir(nested) and len(os.listdir(nested)) >= min_files:
        return nested
    return None


def _free_gb(path):
    import shutil as _sh
    probe = path
    while probe and not os.path.isdir(probe):
        probe = os.path.dirname(probe)
    try:
        return _sh.disk_usage(probe or "/").free / 1e9
    except Exception:
        return float("nan")


def stage_scenario(ctx):
    """Make `deepmimo.get_available_scenarios()` report o1_28.

    The extracted scenario is ~21 GB and can never fit Kaggle's 20 GB
    /kaggle/working quota, so it is kept on the root filesystem and the
    library is pointed at it. See execution guide 0.5.
    """
    args = ctx.args
    deepmimo = _configure_deepmimo(args)
    scen_dir = args.scenario_dir

    print("deepmimo %s | scenarios dir: %s (%.1f GB free)"
          % (deepmimo.__version__, scen_dir, _free_gb(scen_dir)))
    avail = list(deepmimo.get_available_scenarios() or [])
    print("available:", avail)
    if SCENARIO in avail:
        return {"source": "already available"}

    # An extracted copy may survive from an earlier session or from a download
    # whose cross-device move failed (copytree raises before rmtree(src)).
    link = os.path.join(scen_dir, SCENARIO)
    for cand in (os.path.join(scen_dir, SCENARIO),
                 "/root/deepmimo_scenarios/" + SCENARIO,
                 "/kaggle/working/deepmimo_scenarios/" + SCENARIO,
                 "/root/" + SCENARIO,
                 os.path.join(os.path.expanduser("~"), SCENARIO)):
        real = _is_scenario_dir(cand)
        if not real:
            continue
        print("found %d files at %s" % (len(os.listdir(real)), real))
        if os.path.abspath(real) != os.path.abspath(link):
            os.makedirs(scen_dir, exist_ok=True)
            if os.path.exists(link) or os.path.islink(link):
                os.remove(link) if os.path.islink(link) else None
            if not os.path.exists(link):
                os.symlink(real, link)
                print("symlinked %s -> %s" % (link, real))
        if SCENARIO in (deepmimo.get_available_scenarios() or []):
            return {"source": real}

    if not args.download:
        raise SystemExit(
            "\no1_28 is not available and no extracted copy was found.\n"
            "Re-run with --download to fetch it (2.28 GB zip, ~21 GB "
            "extracted, 10-15 min).\n"
            "Do this ONCE -- repeated calls risk an HTTP 429 rate limit.")

    # Download into a SEPARATE directory on the SAME filesystem as scen_dir.
    #   * same filesystem  -> the library's final shutil.move() is a rename,
    #     not a 21 GB copytree, so no cross-device error and no quota blowup
    #   * separate directory -> output_dir == scenarios_dir would make that
    #     final move land inside its own target as <dir>/o1_28/o1_28
    dl_dir = os.path.join(os.path.dirname(scen_dir.rstrip("/\\")) or "/",
                          "dm_download")
    need = 24.0
    free = _free_gb(dl_dir)
    print("downloading %s -> %s (%.1f GB free; ~%.0f GB needed)"
          % (SCENARIO, dl_dir, free, need))
    if free < need:
        raise SystemExit(
            "only %.1f GB free where the scenario must be staged; ~%.0f GB is "
            "needed (2.28 GB zip + 21 GB extracted). Point --scenario-dir at a "
            "bigger filesystem." % (free, need))

    os.makedirs(dl_dir, exist_ok=True)
    os.makedirs(scen_dir, exist_ok=True)
    try:
        # Single attempt. Never retry in a loop -- that is how you earn a 429.
        deepmimo.download(SCENARIO, output_dir=dl_dir)
    except Exception as e:
        print("download() raised: %r" % (e,))
        print("If it got as far as extracting, the files usually survive; "
              "this stage will find them below or on a re-run.")

    for cand in (os.path.join(scen_dir, SCENARIO), os.path.join(dl_dir, SCENARIO)):
        real = _is_scenario_dir(cand)
        if real and not os.path.exists(link):
            os.symlink(real, link)
            print("symlinked %s -> %s" % (link, real))

    avail = list(deepmimo.get_available_scenarios() or [])
    if SCENARIO not in avail:
        raise SystemExit(
            "still not available after download. available=%s\n"
            "Inspect %s and %s, then re-run this stage." % (avail, scen_dir, dl_dir))

    # Reclaim the 2.28 GB zip now that the scenario is in place.
    zp = os.path.join(dl_dir, "%s_downloaded.zip" % SCENARIO)
    if os.path.isfile(zp):
        try:
            os.remove(zp)
            print("removed %s (reclaimed 2.3 GB)" % zp)
        except OSError as e:
            print("could not remove %s: %r" % (zp, e))
    return {"source": "downloaded"}


def stage_data(ctx):
    """Phase 1 -- channels, dead-user filtering, 500/2000 split."""
    from data.preprocess import load_raw, preprocess

    _configure_deepmimo(ctx.args)      # fresh process => config is back to default

    H_all, pos_all = load_raw(n_subsample=ctx.args.subsample)
    info = preprocess(H_all, pos_all, out_dir=ctx.data_dir)

    ctx.state.record(
        n_users_generated=info["n_total"],
        n_dropped_dead=info["n_dropped"],
        n_retained=info["n_retained"],
        pos_min=info["pos_min"], pos_max=info["pos_max"],
        raw_channel_shape=list(H_all.shape),
    )
    return info


def stage_labels(ctx):
    """Phase 2 -- codebook, labels, Figure 1, majority-class baseline."""
    import torch
    from utils.codebook import label_batch
    from evaluation.make_figures import fig_label_hist

    cb = ctx.codebook
    seed_labels = label_batch(ctx.seed_csi, cb)
    test_labels = label_batch(ctx.test_csi, cb)
    torch.save(cb, os.path.join(ctx.data_dir, "codebook.pt"))
    torch.save(seed_labels, os.path.join(ctx.data_dir, "seed_labels.pt"))
    torch.save(test_labels, os.path.join(ctx.data_dir, "test_labels.pt"))
    ctx._cache["seed_labels"] = seed_labels
    ctx._cache["test_labels"] = test_labels
    ctx._cache.pop("majority", None)

    maj, maj_cls = ctx.majority
    allk = torch.cat([seed_labels, test_labels])
    f1 = fig_label_hist(allk, maj, plot_dir=ctx.plot_dir)

    print("\nMAJORITY-CLASS BASELINE: %.4f (class %d)" % (maj, maj_cls))
    ctx.state.record(majority_class_acc=maj, majority_class_id=maj_cls,
                     n_classes_populated=f1["n_populated"],
                     label_counts=f1["counts"])

    # Sanity gate: a degenerate label distribution means the codebook math or
    # the antenna axis is wrong, and everything downstream would be garbage.
    if f1["n_populated"] < 10 and not ctx.args.force:
        raise SystemExit(
            "LABEL GATE FAILED: only %d of 32 beam classes are populated.\n"
            "Expected >~10. Check the antenna axis orientation in "
            "_normalize_channel_shape() and the conjugation in beam_gains().\n"
            "Override with --force only if you have decided this is real."
            % f1["n_populated"])
    return {"majority": maj, "n_populated": f1["n_populated"]}


def _missing_runs(csv_path, ratios, seeds):
    """(ratio, seed) pairs not yet in the CSV -- this is what makes resume work."""
    done = set()
    if os.path.isfile(csv_path):
        import pandas as pd
        try:
            df = pd.read_csv(csv_path)
            done = {(int(r), int(s)) for r, s in
                    zip(df["synthetic_ratio"], df["seed"])}
        except Exception as e:
            print("could not read %s (%r); treating as empty" % (csv_path, e))
    return {r: [s for s in seeds if (r, s) not in done] for r in ratios}


def _run_ratios(ctx, ratios, need_generator):
    """Run only the (ratio, seed) pairs that are still missing."""
    from evaluation.run_ablation import run_sweep

    todo = _missing_runs(ctx.csv_path, ratios, ctx.args.seeds)
    pending = {r: s for r, s in todo.items() if s}
    if not pending:
        print("all requested runs already in %s -- nothing to do" % ctx.csv_path)
        return []

    print("pending runs:", {("R%d" % r): s for r, s in pending.items()})
    G = ctx.generator if need_generator else None
    rows = []
    for ratio, seeds in pending.items():
        rows += run_sweep(
            ctx.seed_csi, ctx.seed_labels, ctx.test_csi, ctx.test_labels,
            ctx.codebook,
            G=G, seed_pos=ctx.seed_pos, test_norms=ctx.test_norm,
            ratios=[ratio], seeds=seeds, csv_path=ctx.csv_path,
            device=ctx.args.device,
            epochs=ctx.args.epochs, n_val=ctx.args.n_val,
        )
    return rows


def stage_baseline(ctx):
    """Phase 3 -- R0 real-only, 3 seeds. THE PROJECT GATE."""
    _run_ratios(ctx, [0], need_generator=False)

    import pandas as pd
    df = pd.read_csv(ctx.csv_path)
    r0 = df[df["synthetic_ratio"] == 0]["top1_acc"]
    maj = ctx.majority[0]
    print("\nR0 top-1: %.4f +/- %.4f (n=%d) | majority class %.4f"
          % (r0.mean(), r0.std(), len(r0), maj))
    ctx.state.record(r0_top1_mean=float(r0.mean()), r0_top1_std=float(r0.std()),
                     r0_n_runs=int(len(r0)))

    if r0.mean() < maj + GATE_MARGIN and not ctx.args.force:
        raise SystemExit(
            "\nBASELINE GATE FAILED: R0 top-1 %.4f does not clearly beat the "
            "majority-class baseline %.4f (needs > %.4f).\n"
            "The classifier has learned little beyond the class prior. No "
            "amount of GAN work fixes this -- debug the labels, the data, or "
            "the training loop first.\n"
            "Override with --force only to deliberately record a null result."
            % (r0.mean(), maj, maj + GATE_MARGIN))
    print("BASELINE GATE PASSED\n")
    return {"r0_mean": float(r0.mean())}


def stage_gan(ctx):
    """Phase 5 -- WGAN-GP. Resumes mid-training from the newest checkpoint."""
    from training.train_gan import train_gan

    t0 = time.time()
    G, D, w_hist = train_gan(
        ctx.seed_csi, ctx.seed_pos,
        n_iters=ctx.args.n_iters, seed=42,
        ckpt_dir=ctx.ckpt_dir, device=ctx.args.device,
    )
    ctx._cache["G"] = G.eval()
    final_w = [h["w_est"] for h in w_hist[-100:]]
    ctx.state.record(gan_iters=ctx.args.n_iters,
                     gan_train_seconds=round(time.time() - t0),
                     gan_final_w_est_mean=float(sum(final_w) / max(len(final_w), 1)))
    return {"iters": len(w_hist)}


def stage_quality(ctx):
    """Phase 6 -- realism checks. test3 is the headline number."""
    from training.train_gan import generate_synthetic
    from evaluation import gan_quality_check as qc

    n = ctx.seed_csi.shape[0]
    synth, _ = generate_synthetic(ctx.generator, ctx.seed_pos, n, seed=0,
                                  device=ctx.args.device)

    qc.test1_heatmaps(ctx.seed_csi, synth, plot_dir=ctx.plot_dir)
    t2 = qc.test2_power_profile(ctx.seed_csi, synth, plot_dir=ctx.plot_dir)
    t3 = qc.test3_discriminability(ctx.seed_csi, synth, device=ctx.args.device,
                                   **({} if ctx.args.qc_epochs is None
                                      else {"epochs": ctx.args.qc_epochs}))
    t4 = qc.test4_label_consistency(ctx.seed_labels, synth, ctx.codebook,
                                    plot_dir=ctx.plot_dir)

    # A high discriminability score is a legitimate finding: it explains weak
    # augmentation gains. Report it, never hide it.
    ctx.state.record(
        gan_discriminability_mean=t3["mean"], gan_discriminability_std=t3["std"],
        gan_discriminability_verdict=t3["verdict"],
        gan_label_overlap=t4["overlap"],
        gan_synth_classes_populated=t4["n_populated_synth"],
        gan_power_wasserstein=t2["wasserstein_power_spread"],
    )
    return {"discriminability": t3["mean"], "label_overlap": t4["overlap"]}


def stage_mechanism(ctx):
    """Why the high-ratio degradation happens -- rank spectrum + TSTR.

    Depends only on the generator, not on the sweep, so it can run after the
    fact without touching ablation_results.csv. TSTR writes its own CSV.
    """
    from training.train_gan import generate_synthetic
    from evaluation import mechanism_check as mc

    n = ctx.seed_csi.shape[0]
    synth, _ = generate_synthetic(ctx.generator, ctx.seed_pos, n, seed=0,
                                  device=ctx.args.device)

    spec = mc.rank_spectrum(ctx.seed_csi, synth, plot_dir=ctx.plot_dir)
    tstr = mc.run_tstr(
        ctx.seed_pos, ctx.test_csi, ctx.test_labels, ctx.codebook,
        ctx.generator, test_norms=ctx.test_norm, seeds=ctx.args.seeds,
        device=ctx.args.device, results_dir=ctx.results_dir,
        epochs=ctx.args.epochs,
    )

    r0 = ctx.state.num("r0_top1_mean")
    extra = {}
    if r0 is not None:
        extra = mc.summarise(spec, tstr, ctx.majority[0], r0)

    ctx.state.record(
        rank90_real=spec["rank90_real"], rank90_synth=spec["rank90_synth"],
        effrank_real=spec["effrank_real"], effrank_synth=spec["effrank_synth"],
        rank_verdict=spec["rank_verdict"],
        tstr_top1_mean=tstr["tstr_top1_mean"],
        tstr_top1_std=tstr["tstr_top1_std"],
        tstr_top3_mean=tstr["tstr_top3_mean"],
        tstr_synth_train_classes=tstr["tstr_synth_train_classes"],
        **extra
    )
    return {"rank90_real": round(spec["rank90_real"], 2),
            "rank90_synth": round(spec["rank90_synth"], 2),
            "tstr_top1": round(tstr["tstr_top1_mean"], 4)}


def stage_sweep(ctx):
    """Phase 7 -- the remaining ratios. Resumes per (ratio, seed)."""
    ratios = [r for r in ctx.args.ratios if r != 0]
    _run_ratios(ctx, ratios, need_generator=True)
    return {"ratios": ratios}


def stage_figures(ctx):
    """Phase 8 -- aggregate, Figures 2 and 5, Table III, significance."""
    from evaluation.run_ablation import aggregate
    from evaluation import make_figures as mf

    df, agg = aggregate(csv_path=ctx.csv_path)
    maj = ctx.majority[0]

    print(agg.to_string(index=False))
    mf.fig_gan_training_curve(ctx.w_hist, plot_dir=ctx.plot_dir,
                              table_dir=ctx.results_dir)
    mf.fig_ablation_curve(agg, maj, plot_dir=ctx.plot_dir)
    mf.table_main_results(agg, maj, table_dir=ctx.results_dir)
    sig = mf.significance(df)

    ctx.state.record(best_ratio=sig["best_ratio"],
                     best_top1_mean=sig["best_mean"],
                     improvement_pp=sig["improvement_pp"],
                     mannwhitney_p=sig["mannwhitney_p"],
                     ttest_p=sig["ttest_p"],
                     n_runs_total=int(len(df)))
    agg.to_csv(os.path.join(ctx.results_dir, "ablation_aggregate.csv"),
               index=False)
    return sig


STAGE_KEYS = {
    "data": ["raw_channel_shape", "n_users_generated", "n_dropped_dead",
             "n_retained", "pos_min", "pos_max"],
    "labels": ["majority_class_acc", "majority_class_id",
               "n_classes_populated", "label_counts"],
    "baseline": ["r0_top1_mean", "r0_top1_std", "r0_n_runs"],
    "gan": ["gan_iters", "gan_train_seconds", "gan_final_w_est_mean"],
    "quality": ["gan_discriminability_mean", "gan_discriminability_std",
                "gan_discriminability_verdict", "gan_label_overlap",
                "gan_synth_classes_populated", "gan_power_wasserstein"],
    "mechanism": ["rank90_real", "rank90_synth", "effrank_real",
                  "effrank_synth", "rank_verdict", "tstr_top1_mean",
                  "tstr_top1_std", "tstr_top3_mean",
                  "tstr_synth_train_classes", "tstr_above_floor_pp"],
    "figures": ["best_ratio", "best_top1_mean", "improvement_pp",
                "mannwhitney_p", "ttest_p", "n_runs_total",
                "trend_spearman_rho", "trend_spearman_p",
                "trend_slope_pp_per_100", "trend_slope_p",
                "low_ratio_mean", "high_ratio_mean", "low_vs_high_pp",
                "low_vs_high_welch_p", "low_vs_high_mw_p",
                "low_vs_high_cohen_d"],
}


def _reset_from(ctx, stage):
    """Invalidate `stage` and everything after it, artifacts included.

    Needed whenever the code a stage depends on changes. The GAN case is the
    dangerous one: editing the generator's final activation does not rename any
    parameter, so a stale checkpoint still load_state_dict()s cleanly and
    training silently resumes the old model. Deleting the checkpoints is the
    only way to be sure a fix actually took effect.
    """
    import shutil as _sh
    if stage not in STAGES:
        raise SystemExit("unknown stage %r; valid: %s" % (stage, STAGES))
    kill = STAGES[STAGES.index(stage):]
    print("resetting:", ", ".join(kill))

    if "gan" in kill and os.path.isdir(ctx.ckpt_dir):
        n = len([f for f in os.listdir(ctx.ckpt_dir) if f.endswith(".pt")])
        _sh.rmtree(ctx.ckpt_dir)
        os.makedirs(ctx.ckpt_dir, exist_ok=True)
        print("  deleted %d GAN checkpoint(s)" % n)

    # The results CSV is shared by baseline (ratio 0) and sweep (the rest), so
    # drop only the rows the reset actually invalidates.
    if os.path.isfile(ctx.csv_path):
        if "baseline" in kill:
            os.remove(ctx.csv_path)
            print("  deleted %s (all runs)" % ctx.csv_path)
        elif "sweep" in kill:
            import pandas as pd
            df = pd.read_csv(ctx.csv_path)
            keep = df[df["synthetic_ratio"] == 0]
            print("  dropped %d synthetic-ratio run(s), kept %d R0 run(s)"
                  % (len(df) - len(keep), len(keep)))
            keep.to_csv(ctx.csv_path, index=False)

    ctx.state.d["done"] = [s for s in ctx.state.d["done"] if s not in kill]
    for s in kill:
        for k in STAGE_KEYS.get(s, []):
            ctx.state.d["numbers"].pop(k, None)
    ctx.state.save()
    ctx._cache.clear()


HANDLERS = {
    "scenario": stage_scenario, "data": stage_data, "labels": stage_labels,
    "baseline": stage_baseline, "gan": stage_gan, "quality": stage_quality,
    "mechanism": stage_mechanism,
    "sweep": stage_sweep, "figures": stage_figures,
}


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------

PAPER_NUMBERS = [
    ("Phase 1 -- data", ["raw_channel_shape", "n_users_generated",
                         "n_dropped_dead", "n_retained", "pos_min", "pos_max"]),
    ("Phase 2 -- labels", ["n_classes_populated", "majority_class_acc",
                           "majority_class_id"]),
    ("Phase 3 -- baseline", ["r0_top1_mean", "r0_top1_std", "r0_n_runs"]),
    ("Phase 5 -- GAN", ["gan_iters", "gan_train_seconds",
                        "gan_final_w_est_mean"]),
    ("Phase 6 -- GAN quality", ["gan_discriminability_mean",
                                "gan_discriminability_std",
                                "gan_discriminability_verdict",
                                "gan_label_overlap",
                                "gan_synth_classes_populated"]),
    ("Phase 6b -- mechanism", ["rank90_real", "rank90_synth",
                               "effrank_real", "effrank_synth",
                               "rank_verdict", "tstr_top1_mean",
                               "tstr_top1_std", "tstr_top3_mean",
                               "tstr_synth_train_classes",
                               "tstr_above_floor_pp"]),
    ("Phase 8 -- PRIMARY trend", ["trend_spearman_rho", "trend_spearman_p",
                                  "trend_slope_pp_per_100",
                                  "trend_slope_p", "n_runs_total"]),
    ("Phase 8 -- post-hoc low vs high", ["low_ratio_mean", "high_ratio_mean",
                                         "low_vs_high_pp",
                                         "low_vs_high_welch_p",
                                         "low_vs_high_mw_p",
                                         "low_vs_high_cohen_d"]),
    ("Phase 8 -- legacy best vs R0 (do NOT headline)",
     ["best_ratio", "best_top1_mean", "improvement_pp",
      "mannwhitney_p", "ttest_p"]),
]


def print_status(ctx):
    st = ctx.state
    print("\n" + "=" * 68)
    print("STAGES")
    for s in STAGES:
        print("  [%s] %s" % ("x" if st.is_done(s) else " ", s))

    print("\nNUMBERS FOR THE PAPER  (state: %s)" % st.path)
    for section, keys in PAPER_NUMBERS:
        vals = [(k, st.num(k)) for k in keys]
        if all(v is None for _, v in vals):
            continue
        print("\n  " + section)
        for k, v in vals:
            if isinstance(v, float):
                v = round(v, 6)
            print("    %-28s %s" % (k, "-- not yet --" if v is None else v))

    if os.path.isfile(ctx.csv_path):
        import pandas as pd
        df = pd.read_csv(ctx.csv_path)
        print("\n  %d ablation runs logged in %s" % (len(df), ctx.csv_path))
        pend = {("R%d" % r): s for r, s in
                _missing_runs(ctx.csv_path, ctx.args.ratios,
                              ctx.args.seeds).items() if s}
        print("  still pending: %s" % (pend if pend else "none"))
    print("=" * 68 + "\n")


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Run the GAN-augmented beam-prediction study end to end.")
    p.add_argument("--work-dir", default="/kaggle/working",
                   help="root for data/, checkpoints/, results/ and state.json")
    p.add_argument("--only", default=None,
                   help="run just these stages (comma-separated): " + ",".join(STAGES))
    p.add_argument("--from", dest="from_stage", default=None,
                   help="start at this stage and run to the end")
    p.add_argument("--redo", default=None,
                   help="comma-separated stages to re-run even if already done")
    p.add_argument("--reset", default=None,
                   help="invalidate this stage and all later ones, deleting the "
                        "artifacts they produced (GAN checkpoints, the rows they "
                        "contributed to the results CSV), then run from there. "
                        "Use this after changing code a finished stage depends on.")
    p.add_argument("--status", action="store_true",
                   help="print progress and recorded numbers, then exit")
    p.add_argument("--download", action="store_true",
                   help="allow a one-shot 2.28 GB scenario download if needed")
    p.add_argument("--force", action="store_true",
                   help="proceed past a failed sanity gate (records a null result)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--n-iters", type=int, default=3000,
                   help="GAN iterations (frozen at 3000 for the paper)")
    p.add_argument("--scenario-dir", default="/root/deepmimo_scenarios",
                   help="absolute dir holding the extracted scenario. Must NOT "
                        "be under /kaggle/working -- o1_28 is ~21 GB and the "
                        "working quota is 20 GB.")
    p.add_argument("--subsample", type=int, default=6000,
                   help="UEs to generate channels for before dead-user filtering")
    p.add_argument("--ratios", default="0,25,50,75,100,150,200")
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--dry-run", action="store_true",
                   help="tiny epochs/iters into <work>/dryrun -- PIPELINE TEST "
                        "ONLY. Never report a number produced this way.")
    a = p.parse_args(argv)

    a.ratios = [int(x) for x in a.ratios.split(",") if x.strip() != ""]
    a.seeds = [int(x) for x in a.seeds.split(",") if x.strip() != ""]
    a.epochs, a.n_val = None, None          # None => the frozen recipe
    a.qc_epochs = None

    if a.dry_run:
        # Isolated work dir: a dry run must never contaminate the real CSV.
        a.work_dir = os.path.join(a.work_dir, "dryrun")
        a.epochs, a.n_val = 2, 20
        a.n_iters = min(a.n_iters, 30)
        a.qc_epochs = 2
        a.ratios = [r for r in a.ratios if r in (0, 100)] or [0, 100]
        a.seeds = a.seeds[:1]
        print("*** DRY RUN -- results go to %s and are NOT publishable ***"
              % a.work_dir)
    return a


def main(argv=None):
    args = parse_args(argv)
    state = State(os.path.join(args.work_dir, "state.json"))
    ctx = Ctx(args, state)
    for d in (ctx.data_dir, ctx.ckpt_dir, ctx.plot_dir):
        os.makedirs(d, exist_ok=True)

    if args.status:
        print_status(ctx)
        return 0

    if args.reset:
        _reset_from(ctx, args.reset)

    if args.only:
        plan = [s.strip() for s in args.only.split(",")]
    elif args.from_stage:
        if args.from_stage not in STAGES:
            raise SystemExit("unknown stage %r" % args.from_stage)
        plan = STAGES[STAGES.index(args.from_stage):]
    else:
        plan = list(STAGES)
    unknown = [s for s in plan if s not in HANDLERS]
    if unknown:
        raise SystemExit("unknown stage(s) %s; valid: %s" % (unknown, STAGES))

    redo = set(s.strip() for s in (args.redo or "").split(",") if s.strip())
    explicit = bool(args.only)      # --only always re-runs what you asked for

    print("plan:", " -> ".join(plan))
    for stage in plan:
        if state.is_done(stage) and stage not in redo and not explicit:
            print("\n== %s: already done, skipping ==" % stage)
            continue
        print("\n" + "=" * 68)
        print("== %s ==" % stage)
        print("=" * 68)
        t0 = time.time()
        try:
            out = HANDLERS[stage](ctx)
        except SystemExit:
            raise
        except Exception:
            traceback.print_exc()
            print_status(ctx)
            raise SystemExit(
                "\n%s FAILED after %.0fs (traceback above). Earlier stages are "
                "saved -- fix the cause and re-run the same command; finished "
                "stages will be skipped." % (stage, time.time() - t0))
        state.finish(stage)
        print("-- %s done in %.0fs: %s" % (stage, time.time() - t0, out))

    print_status(ctx)
    print("Remember: save /kaggle/working/ to a Kaggle Dataset before the "
          "session ends, or everything above is lost.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
