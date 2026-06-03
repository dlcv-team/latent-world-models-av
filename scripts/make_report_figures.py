"""Generate DiT-WAM report/supplementary data-plot figures from committed JSON.

LOCAL matplotlib (trivial / seconds, no GPU). Every figure is traceable to a
committed artifact in artifacts/full/. Retained + reproducible.

Usage:
  python3 make_report_figures.py --figs F6,F7,F8            # P0 gate
  python3 make_report_figures.py --figs S1,S2,S3,S4,S5,S6,S7 # P1 supplementary
  python3 make_report_figures.py --all --profile poster
Honesty rules baked in: error bars labeled (kid_std / 3-seed std); F7 "inspired by
Blau-Michaeli" (not Pareto); S3-S7 tagged Preliminary; splits stated in titles.
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
import viz_style as S

SEED_FILES = {
    0: "fid_eval_full600.json",
    1: "fid_eval_vol_dits_vae_latent_diffusion_h16_seed_1_dit.pt.json",
    2: "fid_eval_vol_dits_vae_latent_diffusion_h16_seed_2_dit.pt.json",
}
METHODS = [("direct", "Direct\n(regression)", "direct"),
           ("diffusion_raw", "Diffusion\n(raw)", "diffusion"),
           ("diffusion_calib", "Diffusion\n(train-calib)", "calib"),
           ("vae_gt", "VAE-GT\n(ceiling)", "ceiling")]


def _kidfid(seed, horizon, method, metric):
    d = S.load(SEED_FILES[seed])
    h = str(horizon)
    if h not in d["by_horizon"]:
        return None
    return d["by_horizon"][h][method][metric]


def f6(profile):
    """F6: FID/KID distribution realism. t+4 = 3-seed mean+/-std; t+16 = seed0 only."""
    S.apply(profile)
    fig, axes = plt.subplots(1, 2, figsize=S.figsize(profile, 7.0, 2.9))
    horizons = [(3, "t+4"), (15, "t+16")]
    width = 0.19
    for ax, metric, ylab in [(axes[0], "kid_mean", "KID  (lower = closer to real)"),
                             (axes[1], "fid", "FID  (lower = closer to real)")]:
        for mi, (mk, mlab, ck) in enumerate(METHODS):
            xs, ys, es = [], [], []
            for hi, (h, _) in enumerate(horizons):
                vals = [_kidfid(s, h, mk, metric) for s in (0, 1, 2)]
                vals = [v for v in vals if v is not None]
                xs.append(hi + (mi - 1.5) * width)
                ys.append(float(np.mean(vals)))
                es.append(float(np.std(vals)) if len(vals) > 1 else 0.0)
            ax.bar(xs, ys, width, yerr=es, capsize=2.5, color=S.C[ck],
                   label=mlab.replace("\n", " "), edgecolor="white", linewidth=0.5)
        ax.set_xticks(range(len(horizons)))
        ax.set_xticklabels([h for _, h in horizons])
        ax.set_ylabel(ylab)
    # ratio annotation (t+16, KID): direct vs diff-calib
    dr = _kidfid(0, 15, "direct", "kid_mean"); cc = _kidfid(0, 15, "diffusion_calib", "kid_mean")
    axes[0].set_title(f"KID $\\downarrow$   (train-calib $\\approx${dr/cc:.1f}$\\times$ lower than regression)")
    axes[1].set_title("FID $\\downarrow$")
    axes[0].legend(frameon=False, ncol=2, loc="upper left", fontsize=plt.rcParams["legend.fontsize"]-1)
    fig.suptitle("Distribution realism: diffusion wins where regression's blur fails "
                 "(600 test windows; t+4 = 3 seeds, t+16 = seed 0)", y=1.04,
                 fontsize=plt.rcParams["axes.titlesize"])
    return S.savefig(fig, "fig_fidkid", profile)


def f7(profile):
    """F7: distortion-perception frontier (frontier_eval.json ONLY)."""
    S.apply(profile)
    d = S.load("frontier_eval.json")["points"]
    fig, ax = plt.subplots(figsize=S.figsize(profile, 3.4, 3.0))
    pts = [("direct", "Direct (regression mean)", "direct", True),
           ("diff_w1", "Diffusion (raw, w=1)", "diffusion", False),
           ("interp_0.5", "Interp $\\alpha{=}0.5$", "interp", True),
           ("diff_caltrain", "Diffusion + train-calib", "calib", True)]
    for key, lab, ck, strong in pts:
        p = d[key]
        ax.errorbar(p["cossim"], p["kid_mean"], yerr=p["kid_std"], fmt="o",
                    color=S.C[ck], alpha=1.0 if strong else 0.45, capsize=3,
                    label=lab, zorder=3, markeredgecolor="white", markeredgewidth=0.6)
    # frontier guide line through the 3 strong operating points
    strong_pts = sorted([(d[k]["cossim"], d[k]["kid_mean"]) for k, _, _, s in pts if s])
    ax.plot([x for x, _ in strong_pts], [y for _, y in strong_pts], "--",
            color="#777777", lw=1.0, zorder=1)
    ax.set_xlabel("CosSim to GT  $\\rightarrow$ better distortion")
    ax.set_ylabel("KID  $\\downarrow$ better perception")
    ax.set_title("Empirical distortion–perception frontier\n(inspired by Blau–Michaeli, t+16, 150 windows)")
    ax.legend(frameon=False, loc="upper center", fontsize=plt.rcParams["legend.fontsize"]-1)
    ax.margins(0.18)
    return S.savefig(fig, "fig_frontier", profile)


def f8(profile):
    """F8: action controllability Spearman(steer->shift)."""
    S.apply(profile)
    d = S.load("controllability.json")
    fig, ax = plt.subplots(figsize=S.figsize(profile, 3.2, 3.0))
    labs = ["Direct\n(regression)", "Diffusion\n(ours)"]
    rhos = [d["direct"]["mean_spearman_steer_to_shift"], d["diffusion"]["mean_spearman_steer_to_shift"]]
    nval = [d["direct"]["n_valid"], d["diffusion"]["n_valid"]]
    frac = [d["direct"]["frac_monotone_correct"], d["diffusion"]["frac_monotone_correct"]]
    cols = [S.C["direct"], S.C["diffusion"]]
    bars = ax.bar(labs, rhos, color=cols, edgecolor="white", width=0.6)
    ax.axhline(0, color="#444444", lw=0.8)
    ax.set_ylabel("Spearman($\\rho$): steer $\\rightarrow$ scene shift")
    ax.set_ylim(-0.45, 1.25)
    for b, r, n, fr in zip(bars, rhos, nval, frac):
        ytxt = r + 0.04 if r > 0 else 0.05   # negative bar: annotate just above the zero line (no bottom clip)
        ax.text(b.get_x() + b.get_width()/2, ytxt,
                f"$\\rho$={r:+.2f}\n{int(fr*100)}% sign-correct\n(n={n})", ha="center",
                va="bottom", fontsize=plt.rcParams["legend.fontsize"]-1)
    ax.set_title("Action controllability: steering controls the\ngenerated scene shift for diffusion, not regression", pad=10)
    return S.savefig(fig, "fig_controllability", profile)


def s1(profile):
    """S1: temporal motion fraction of GT."""
    S.apply(profile)
    d = S.load("motion_eval.json")
    fig, ax = plt.subplots(figsize=S.figsize(profile, 3.0, 2.8))
    vals = [d["direct_frac_of_gt"]*100, d["diffusion_frac_of_gt"]*100]
    ax.bar(["Direct\n(regression)", "Diffusion\n(ours)"], vals, color=[S.C["direct"], S.C["diffusion"]],
           edgecolor="white", width=0.6)
    ax.axhline(100, ls="--", color=S.C["ceiling"], lw=1.0); ax.text(1.45, 100, "GT", va="center", color=S.C["ceiling"])
    for i, v in enumerate(vals): ax.text(i, v+2, f"{v:.0f}%", ha="center")
    ax.set_ylabel("% of GT inter-frame change"); ax.set_ylim(0, 115)
    ax.set_title("Temporal dynamics: diffusion is more dynamic\nthan the static-ish blur (150 windows)")
    return S.savefig(fig, "fig_motion", profile)


def s2(profile):
    """S2: representation-uncertainty regime axis (copy-CosSim predictability)."""
    S.apply(profile)
    fig, ax = plt.subplots(figsize=S.figsize(profile, 4.6, 2.4))
    pts = [(0.85, "Pooled semantic", "MLP / direct"), (0.76, "Spatial semantic\n(ViT/DINO)", "direct DiT > MLP"),
           (0.43, "VAE / pixel", "diffusion-for-realism")]
    ax.axvspan(0.0, 0.6, color=S.C["diffusion"], alpha=0.08)
    ax.text(0.3, 0.62, "diffusion helps\n(high uncertainty)", ha="center", color=S.C["diffusion"], fontsize=plt.rcParams["legend.fontsize"]-1)
    for x, lab, regime in pts:
        ax.scatter([x], [0.5], s=120 if profile=="report" else 600, color=S.C["diffusion"] if x < 0.6 else S.C["direct"], zorder=3, edgecolor="white")
        ax.annotate(f"{lab}\n({regime})", (x, 0.5), (x, 0.78), ha="center", fontsize=plt.rcParams["legend.fontsize"]-1,
                    arrowprops=dict(arrowstyle="-", color="#999"))
    ax.set_xlim(0.3, 0.95); ax.set_ylim(0.3, 0.95); ax.set_yticks([])
    ax.set_xlabel("Future predictability  (copy-baseline CosSim) $\\rightarrow$")
    ax.set_title("Representation–uncertainty axis: the diffusion advantage tracks the regime flip")
    return S.savefig(fig, "fig_regime_axis", profile)


def s3(profile):
    """S3: P1 low-end scaling (3.03M vs 5.4M), deployable train-calib @ t+16."""
    S.apply(profile)
    big = S.load("frontier_eval.json")["points"]["diff_caltrain"]
    small = S.load("frontier_eval_nb2_3.0M.json")["points"]["diff_caltrain"]
    params = [3.03, 5.40]
    fig, ax = plt.subplots(figsize=S.figsize(profile, 3.2, 2.8))
    ax.plot(params, [small["kid_mean"], big["kid_mean"]], "o-", color=S.C["calib"], label="KID")
    ax.set_xlabel("DiT parameters (M)"); ax.set_ylabel("KID $\\downarrow$ (deployable train-calib, t+16)")
    ax2 = ax.twinx(); ax2.plot(params, [small["fid"], big["fid"]], "s--", color=S.C["diffusion"], label="FID")
    ax2.set_ylabel("FID $\\downarrow$"); ax2.grid(False)
    ax.set_title("Preliminary: capacity helps in this small-data regime\n(2 points, single seed)")
    l1,la1=ax.get_legend_handles_labels(); l2,la2=ax2.get_legend_handles_labels()
    ax.legend(l1+l2, la1+la2, frameon=False, loc="center right")
    return S.savefig(fig, "fig_scaling", profile)


def s4(profile):
    """S4: P4 CFG control-fidelity. rho(w) [control] vs LPIPS(w) [fidelity]."""
    S.apply(profile)
    w = [1, 2, 3]
    rho = [S.load("controllability.json")["diffusion"]["mean_spearman_steer_to_shift"],
           S.load("controllability_cfg2.0.json")["diffusion"]["mean_spearman_steer_to_shift"],
           S.load("controllability_cfg3.0.json")["diffusion"]["mean_spearman_steer_to_shift"]]
    ge = S.load("gen_eval_direct_diffusion.json")
    blk = ge["models"]["diffusion"]
    lpips = [blk[f"cfg_{float(x):.1f}"]["lpips"]["15"] for x in w]
    fig, ax = plt.subplots(figsize=S.figsize(profile, 3.4, 2.8))
    ax.plot(w, rho, "o-", color=S.C["diffusion"], label="$\\rho$ control (steer→shift)")
    ax.set_xlabel("CFG guidance scale $w$"); ax.set_ylabel("$\\rho$ control $\\uparrow$"); ax.set_xticks(w)
    ax2 = ax.twinx(); ax2.plot(w, lpips, "s--", color=S.C["accent"], label="LPIPS to GT (↓ better)"); ax2.grid(False)
    ax2.set_ylabel("LPIPS $\\downarrow$ fidelity")
    ax.set_title("Preliminary: CFG>1 does not improve control quality;\nfidelity mildly worse. We operate at $w{=}1$.")
    l1,la1=ax.get_legend_handles_labels(); l2,la2=ax2.get_legend_handles_labels()
    ax.legend(l1+l2, la1+la2, frameon=False, loc="center right")
    return S.savefig(fig, "fig_cfg_tradeoff", profile)


def s5(profile):
    """S5: P5 inverse-control MAE vs random-pick chance."""
    S.apply(profile)
    d = S.load("planning_probe.json")
    fig, ax = plt.subplots(figsize=S.figsize(profile, 3.0, 2.8))
    ratio = [d["direct"]["inverse_control"]["mae_vs_chance_ratio"],
             d["diffusion"]["inverse_control"]["mae_vs_chance_ratio"]]
    nval = [d["direct"]["inverse_control"]["n_valid"], d["diffusion"]["inverse_control"]["n_valid"]]
    bars = ax.bar(["Direct\n(regression)", "Diffusion\n(ours)"], ratio, color=[S.C["direct"], S.C["diffusion"]], edgecolor="white", width=0.6)
    ax.axhline(1.0, ls="--", color=S.C["chance"]); ax.text(1.45, 1.0, "chance", va="center", color="#777")
    for b, r, n in zip(bars, ratio, nval):
        ax.text(b.get_x()+b.get_width()/2, r+0.03, f"{r:.2f}×\nn={n}", ha="center")
    ax.set_ylabel("inverse-control error / chance  $\\downarrow$")
    ax.set_title("Preliminary: diffusion recovers a held-out target steer\nbelow chance; regression is worse than chance")
    return S.savefig(fig, "fig_inverse_control", profile)


def s6(profile):
    """S6: P2/P7 rollout 2x2 dFID (compounding + test-time reprojection)."""
    S.apply(profile)
    c = S.load("rollout_eval_feedback.json")["cells"]
    fig, ax = plt.subplots(figsize=S.figsize(profile, 3.4, 2.8))
    groups = [("Direct", "direct"), ("Diffusion", "diffusion")]
    width = 0.36
    for gi, (glab, gk) in enumerate(groups):
        raw = c[f"{gk}_raw"]["dFID_t32_minus_t16"]; rep = c[f"{gk}_vae_reproject"]["dFID_t32_minus_t16"]
        ax.bar(gi-width/2, raw, width, color=S.C[gk], label="raw feedback" if gi == 0 else None, edgecolor="white")
        ax.bar(gi+width/2, rep, width, color=S.C[gk], alpha=0.5, hatch="//", label="VAE re-project" if gi == 0 else None, edgecolor="white")
    ax.set_xticks([0, 1]); ax.set_xticklabels([g for g, _ in groups])
    ax.set_ylabel("$\\Delta$FID  (t+32 $-$ t+16)  $\\downarrow$")
    ax.legend(frameon=False)
    ax.set_title("Preliminary: re-projection stabilizes regression rollout\n(36→18) but not diffusion (98→106)")
    return S.savefig(fig, "fig_rollout", profile)


def s7(profile):
    """S7: P3 frame-rate multimodality spread vs temporal stride."""
    S.apply(profile)
    d = S.load("framerate_preflight.json")["strides"]
    ks = sorted(d.keys(), key=int)
    xs = [d[k]["seconds_ahead"] for k in ks]; ys = [d[k]["sigma_scene_future"] for k in ks]
    fig, ax = plt.subplots(figsize=S.figsize(profile, 3.2, 2.8))
    ax.plot(xs, ys, "o-", color=S.C["diffusion"])
    base = ys[0]
    ax.axhline(base*1.2, ls="--", color=S.C["accent"]); ax.text(xs[-1], base*1.2, " 1.2× gate", va="bottom", color=S.C["accent"], ha="right")
    for x, y in zip(xs, ys): ax.text(x, y+0.004, f"{y:.3f}", ha="center", fontsize=plt.rcParams["legend.fontsize"]-1)
    ax.set_xlabel("effective horizon (s)  [lower frame rate $\\rightarrow$]"); ax.set_ylabel("conditional future spread $\\sigma$")
    ax.set_title(f"Preliminary: no multimodality growth at lower Hz\n(ratio {ys[-1]/ys[0]:.2f}× < 1.2 gate; VAE latents)")
    return S.savefig(fig, "fig_framerate", profile)


def smf(profile):
    """Motion-fidelity (honest): low-freq coherent scene-motion vs high-freq texture, fraction of GT.
    The headline correction to the old 'more dynamic than the blur' claim."""
    S.apply(profile)
    d = S.load("motion_fidelity.json")
    groups = ["all_scenes", "high_motion_quartile"]; glab = ["All scenes", "High-motion\nquartile"]
    dlow = [d[g]["direct"]["lowfreq_frac_of_gt"] for g in groups]
    glow = [d[g]["diffusion"]["lowfreq_frac_of_gt"] for g in groups]
    ghigh = [d[g]["diffusion"]["highfreq_frac_of_gt"] for g in groups]
    x = np.arange(2); w = 0.26
    fig, ax = plt.subplots(figsize=S.figsize(profile, 3.8, 3.0))
    ax.bar(x - w, dlow, w, color=S.C["direct"], edgecolor="white", label="Direct: scene-motion (low-freq)")
    ax.bar(x, glow, w, color=S.C["diffusion"], edgecolor="white", label="Diffusion: scene-motion (low-freq)")
    ax.bar(x + w, ghigh, w, color=S.C["diffusion"], alpha=0.4, hatch="//", edgecolor="white", label="Diffusion: texture (high-freq)")
    ax.axhline(1.0, ls="--", color=S.C["ceiling"], lw=1.0); ax.text(1.48, 1.0, "GT", va="center", color=S.C["ceiling"])
    for xi, v in zip(x - w, dlow): ax.text(xi, v + 0.02, f"{v:.2f}", ha="center", fontsize=plt.rcParams["legend.fontsize"]-1)
    for xi, v in zip(x, glow): ax.text(xi, v + 0.02, f"{v:.2f}", ha="center", fontsize=plt.rcParams["legend.fontsize"]-1)
    for xi, v in zip(x + w, ghigh): ax.text(xi, v + 0.02, f"{v:.2f}", ha="center", fontsize=plt.rcParams["legend.fontsize"]-1)
    ax.set_xticks(x); ax.set_xticklabels(glab); ax.set_ylim(0, 1.15)
    ax.set_ylabel("fraction of GT frame-to-frame change")
    ax.legend(frameon=False, loc="upper center", fontsize=plt.rcParams["legend.fontsize"]-1)
    ax.set_title("Diffusion reproduces GT texture (0.98) but little coherent motion;\nthe blurry mean captures MORE scene motion (40 scenes)")
    return S.savefig(fig, "fig_motion_fidelity", profile)


REG = {"F6": f6, "F7": f7, "F8": f8, "S1": s1, "S2": s2, "S3": s3, "S4": s4, "S5": s5, "S6": s6, "S7": s7, "SMF": smf}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--figs", default="F6,F7,F8")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--profile", default="report", choices=["report", "poster"])
    a = ap.parse_args()
    keys = list(REG) if a.all else [k.strip().upper() for k in a.figs.split(",") if k.strip()]
    print(f"[figs] profile={a.profile} -> {keys}")
    for k in keys:
        print(f"[{k}]"); REG[k](a.profile)
    print("done.")
