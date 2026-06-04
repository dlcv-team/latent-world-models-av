"""Phase-2 qualitative assets + videos for the DiT-WAM project page / poster / teaser.

MODAL A10G (ffmpeg in image). Loads HF/volume seed_0 diffusion+direct + SD-VAE; reuses the
make_figure/controllability/rollout decode patterns. Renders (honest, post motion-finding):
  - f5         : richer MULTI-SCENE 4-row qualitative grid (RGB / VAE-GT / direct-blur / diffusion-sharp)
  - counter    : V1 action-counterfactual -> steer L/S/R future strips (FIXED noise) -> mp4+gif (controllability)
  - steersweep : 9 fixed-noise steer values x scenes -> decoded frames (widget data for the project page slider)
  - rollout    : V2 direct-vs-diffusion t->t+32 side-by-side -> mp4 at low fps (perception win + honest compounding)
  - denoise    : V3 DDIM denoising of a future latent -> mp4
  - diversity  : V4 K samples (different noise) -> grid mp4
  - teaser     : F1 teaser COMPONENTS (present / direct-blur / diffusion-sharp panels) -> PNG for LaTeX assembly
Outputs to /vol/viz/ ; download with `modal volume get nuscenes-full /viz <local>`. Large mp4 -> HF (not git).

Usage: modal run scripts/make_viz_assets_modal.py --which counter,steersweep,f5 --n-scenes 3
"""
from __future__ import annotations
import importlib.util, os
from pathlib import Path
try:
    import modal
except ImportError:
    modal = None

SCRIPT_DIR = Path(__file__).resolve().parent
TRAIN_SCRIPT = SCRIPT_DIR / "train_dit_vae_modal.py"
if modal is not None:
    app = modal.App("lwm-av-viz")
    vol = modal.Volume.from_name("nuscenes-full")
    image = (modal.Image.debian_slim(python_version="3.12")
             .apt_install("ffmpeg")
             # EXPLICIT torch-2.5.1-compatible pins (fresh build pulls latest transformers which references
             # torch.float8_e8m0fnu absent in 2.5.1 -> pin transformers/diffusion stack to late-2024 set)
             .pip_install("torch==2.5.1", "torchvision==0.20.1", "numpy==1.26.4", "Pillow>=10.0",
                          "diffusers==0.31.0", "transformers==4.46.3", "accelerate==1.1.1",
                          "matplotlib>=3.8", "scipy>=1.11", "imageio>=2.34", "imageio-ffmpeg>=0.4.9")
             .add_local_file(str(TRAIN_SCRIPT), remote_path="/root/train_dit_vae_modal.py"))
else:
    app = vol = image = None

VOL_PATH = "/vol"
SPATIAL_DIR = f"{VOL_PATH}/embeddings/spatial"
CKPT_DIR = f"{VOL_PATH}/dits/vae_latent"
VAE_NPZ = f"{SPATIAL_DIR}/sd_vae_latents.npz"
OUT = f"{VOL_PATH}/viz"
SCALING, HORIZON, DIFFUSION_STEPS = 0.18215, 16, 1000
CKPT_PATHS = {"direct": f"{CKPT_DIR}/h{HORIZON}/seed_0/dit.pt",
              "diffusion": f"{CKPT_DIR}/diffusion/h{HORIZON}/seed_0/dit.pt"}


def _decorator(fn):
    if app is not None:
        return app.function(volumes={VOL_PATH: vol}, image=image, gpu="A10G", timeout=7200, memory=32768)(fn)
    return fn


@_decorator
def render(which: str = "counter,steersweep,f5,teaser,rollout", n_scenes: int = 3, seed: int = 0, scene_ids: str = "",
           cfg_w: float = 1.0, winner: str = "sdedit_0.5"):
    import numpy as np, torch
    import torch.nn.functional as F
    import imageio.v2 as imageio
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from diffusers import AutoencoderKL
    from PIL import Image
    import torchvision.transforms.functional as TF

    spec = importlib.util.spec_from_file_location("tv", "/root/train_dit_vae_modal.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    patchify, unpatchify = mod.patchify, mod.unpatchify
    AnchoredVAEDiT, FourierActionEmbedding = mod.AnchoredVAEDiT, mod.FourierActionEmbedding
    DIT_CONFIG, FOURIER_CONFIG = mod.DIT_CONFIG, mod.FOURIER_CONFIG
    PATCH_DIM, N_SPATIAL = mod.PATCH_DIM, mod.N_SPATIAL
    Sched = mod._define_noise_schedule()
    dev = torch.device("cuda")
    torch.set_grad_enabled(False)
    sch = Sched(n_steps=DIFFUSION_STEPS).to(dev); alphas = sch.alphas_cumprod
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(dev).eval()
    os.makedirs(OUT, exist_ok=True)
    data = np.load(VAE_NPZ, allow_pickle=True)
    lat = data["vae_latents"].astype(np.float32); scenes = data["scene_names"]; splits = data["splits"]
    steers = data["steer_norms"].astype(np.float32); accels = data["accel_norms"].astype(np.float32)
    ipaths = data["image_paths"] if "image_paths" in data else None
    tr = np.where(splits == "train")[0]; p5, p95 = float(np.percentile(steers[tr], 5)), float(np.percentile(steers[tr], 95))
    pmid = float(np.percentile(steers[tr], 50))
    te = np.where(splits == "test")[0]
    picks = []
    for sc in np.unique(scenes[te]):
        idx = te[scenes[te] == sc]
        if len(idx) > HORIZON: picks.append(int(idx[0]))
        if len(picks) >= n_scenes: break
    if scene_ids:  # override with mined scene window-indices (M1 scene mining)
        picks = [int(x) for x in scene_ids.split(",") if x.strip()]

    def load(tag):
        ck = torch.load(CKPT_PATHS.get(tag, tag), map_location=dev, weights_only=False)
        dit = AnchoredVAEDiT(horizon=HORIZON, n_spatial=N_SPATIAL, **DIT_CONFIG).to(dev)
        fo = FourierActionEmbedding(**FOURIER_CONFIG).to(dev)
        dit.load_state_dict(ck["dit"]); fo.load_state_dict(ck["fourier"])
        if "ema" in ck and ck["ema"]:
            for nm, p in dit.named_parameters():
                if nm in ck["ema"]: p.data.copy_(ck["ema"][nm].to(dev))
            for nm, p in fo.named_parameters():
                if nm in ck["ema"]: p.data.copy_(ck["ema"][nm].to(dev))
        dit.eval(); fo.eval(); return dit, fo, ck["z_mean"].to(dev), ck["z_std"].to(dev)
    d_dit, d_fo, dzm, dzs = load("direct"); g_dit, g_fo, gzm, gzs = load("diffusion")

    def decode(g): return ((vae.decode(g.clamp(-6, 6) / SCALING).sample.clamp(-1, 1)) + 1) / 2
    def to_u8(t): return (t[0].permute(1, 2, 0).clamp(0, 1).detach().cpu().numpy() * 255).astype(np.uint8)
    def calib(grid, z_t):  # per-channel mean-match to present (visualization calib)
        return grid - grid.mean(dim=(2, 3), keepdim=True) + z_t.mean(dim=(2, 3), keepdim=True)
    def act_const(fi, sv): return torch.tensor(np.stack([[sv, accels[fi + k]] for k in range(HORIZON)]),
                                               dtype=torch.float32, device=dev).unsqueeze(0)
    def rgb_real(n):
        if ipaths is None or n >= len(ipaths): return None
        p = f"{VOL_PATH}/nuscenes/{ipaths[n]}"
        if not os.path.exists(p): return None
        im = Image.open(p).convert("RGB"); w, h = im.size; c = min(w, h)
        im = im.crop(((w-c)//2, (h-c)//2, (w-c)//2+c, (h-c)//2+c)).resize((256, 256))
        return (np.asarray(im)).astype(np.uint8)

    # ---- DDIM + constant-steer action window (shared by single-pass / AR / SDEdit) ----
    NS = 50; STRIDE = max(DIFFUSION_STEPS // NS, 1); TS = list(reversed(list(range(0, DIFFUSION_STEPS, STRIDE))[:NS]))
    def _act_window(base, sv):  # H-step constant-steer action window starting at frame `base` (clamped at sequence end)
        Nf = len(accels)
        return torch.tensor(np.stack([[sv, accels[min(base + k, Nf - 1)]] for k in range(HORIZON)]),
                            dtype=torch.float32, device=dev).unsqueeze(0)
    def _ddim(x, ztn, a, ts, cfg_w):  # DDIM x0 loop; cfg_w>1 guides toward the action (null = zeroed embedding)
        for i, tv in enumerate(ts):
            t = torch.full((1,), tv, device=dev, dtype=torch.long); px0 = g_dit(x, ztn, a, t)
            if cfg_w != 1.0:
                pu = g_dit(x, ztn, torch.zeros_like(a), t); px0 = pu + cfg_w * (px0 - pu)
            at = alphas[tv]; ap = alphas[ts[i + 1]] if i < len(ts) - 1 else torch.tensor(1.0, device=dev)
            nd = (x - torch.sqrt(at) * px0) / torch.sqrt(1 - at + 1e-8); x = torch.sqrt(ap) * px0 + torch.sqrt(1 - ap) * nd
        return x

    # ---- raw-latent grid generators (moving-row bake-off candidates) ----
    def diff_grid(fi, sv, nseed, cfg_w=1.0):  # single-pass diffusion (present anchor + action)
        zt = torch.tensor(lat[fi:fi+1], device=dev); ztn = (patchify(zt)-gzm)/gzs; a = g_fo(act_const(fi, sv))
        gen = torch.Generator(device=dev).manual_seed(nseed); x = torch.randn(1, HORIZON*N_SPATIAL, PATCH_DIM, device=dev, generator=gen)
        return (_ddim(x, ztn, a, TS, cfg_w) * gzs + gzm).reshape(1, HORIZON, N_SPATIAL, PATCH_DIM)
    def direct_grid(fi, sv, gain=1.0):  # C0: direct regression; gain>1 amplifies per-step residual from present
        zt = torch.tensor(lat[fi:fi+1], device=dev); ztn = (patchify(zt)-dzm)/dzs
        zr = ztn.unsqueeze(1).expand(-1, HORIZON, -1, -1).reshape(1, HORIZON*N_SPATIAL, PATCH_DIM)
        dp = (d_dit(zr, ztn, d_fo(act_const(fi, sv)), torch.zeros(1, dtype=torch.long, device=dev))*dzs+dzm).reshape(1, HORIZON, N_SPATIAL, PATCH_DIM)
        if gain != 1.0:
            ztp = patchify(zt).unsqueeze(1)  # (1,1,N_SPATIAL,PATCH_DIM) present in patch space
            dp = ztp + gain * (dp - ztp)
        return dp
    def sdedit_grid(fi, sv, nseed, strength, cfg_w=1.0):  # C2: img2img refine the DIRECT prediction with diffusion
        zt = torch.tensor(lat[fi:fi+1], device=dev); ztn = (patchify(zt)-gzm)/gzs; a = g_fo(act_const(fi, sv))
        x0 = (direct_grid(fi, sv, 1.0).reshape(1, HORIZON*N_SPATIAL, PATCH_DIM) - gzm) / gzs   # direct future, normalized
        k0 = int(min(max(strength, 0.0), 1.0) * (len(TS) - 1)); tv0 = TS[k0]
        gen = torch.Generator(device=dev).manual_seed(nseed)
        x = torch.sqrt(alphas[tv0]) * x0 + torch.sqrt(1 - alphas[tv0]) * torch.randn(x0.shape, device=dev, generator=gen)
        return (_ddim(x, ztn, a, TS[k0:], cfg_w) * gzs + gzm).reshape(1, HORIZON, N_SPATIAL, PATCH_DIM)
    def interp_grid(fi, sv, nseed, beta=0.5, cfg_w=1.0):  # C4: balanced operating point in raw latent space
        return beta * direct_grid(fi, sv, 1.0) + (1 - beta) * diff_grid(fi, sv, nseed, cfg_w)

    # ---- decode wrappers ----
    def grid_frames(grid, zt):  # raw grid -> list of decoded [0,1] tensors, calibrated to present
        return [decode(calib(unpatchify(grid[:, k]), zt)) for k in range(HORIZON)]
    def ar_diffusion_frames(fi, sv, nseed, reproject_fb=False, cfg_w=1.0):  # C1: open-loop chained (re-anchor each step)
        z0 = torch.tensor(lat[fi:fi+1], device=dev); z_curr = z0; frames = []
        for kk in range(HORIZON):
            ztn = (patchify(z_curr)-gzm)/gzs; a = g_fo(_act_window(fi+kk, sv))
            gen = torch.Generator(device=dev).manual_seed(nseed + kk)
            x = torch.randn(1, HORIZON*N_SPATIAL, PATCH_DIM, device=dev, generator=gen)
            z_next = unpatchify((_ddim(x, ztn, a, TS, cfg_w)*gzs+gzm).reshape(1, HORIZON, N_SPATIAL, PATCH_DIM)[:, 0])
            if reproject_fb:
                z_next = vae.encode(decode(z_next) * 2 - 1).latent_dist.mean * SCALING
            frames.append(decode(calib(z_next, z0))); z_curr = z_next
        return frames
    def diff_future(fi, sv, nseed, cfg_w=1.0):
        return [to_u8(f) for f in grid_frames(diff_grid(fi, sv, nseed, cfg_w), torch.tensor(lat[fi:fi+1], device=dev))]
    def direct_future(fi, sv, gain=1.0):  # decoded direct (no calib, matching the diagnostic / existing F5 row)
        return [to_u8(decode(unpatchify(direct_grid(fi, sv, gain)[:, k]))) for k in range(HORIZON)]

    # ---- motion metrics (ported from eval motion_fidelity; automatic bake-off gate) ----
    BLUR_KS, BLUR_SIGMA = 31, 8.0
    def _low(img): return TF.gaussian_blur(img, BLUR_KS, [BLUR_SIGMA, BLUR_SIGMA])
    def _rms(a): return float(a.pow(2).mean().sqrt().item())
    def _seq_lowhigh(fr):
        lo = [_low(f) for f in fr]
        L = float(np.mean([_rms(lo[i+1]-lo[i]) for i in range(len(fr)-1)]))
        H = float(np.mean([_rms((fr[i+1]-lo[i+1])-(fr[i]-lo[i])) for i in range(len(fr)-1)]))
        return L, H
    def _prof(img, ax): return img[0].mean(dim=(0, ax)).detach().cpu().numpy()  # ax=1 horizontal, ax=2 vertical
    def _sh1(p, r):
        a = p - p.mean(); b = r - r.mean(); cc = np.correlate(a, b, "full"); return int(cc.argmax() - (len(p) - 1))
    def _disp_seq(fr): return [(_sh1(_prof(fr[i+1], 1), _prof(fr[i], 1)), _sh1(_prof(fr[i+1], 2), _prof(fr[i], 2))) for i in range(len(fr)-1)]
    def _dir_corr(pr, gt):
        cs = [(px*gx+py*gy)/(((gx*gx+gy*gy)**0.5)*((px*px+py*py)**0.5))
              for (px, py), (gx, gy) in zip(pr, gt) if (gx*gx+gy*gy) > 0 and (px*px+py*py) > 0]
        return float(np.mean(cs)) if cs else 0.0
    def _disp_mag(s): return float(np.mean([(dx*dx+dy*dy)**0.5 for dx, dy in s]))
    def _sharp(img):  # Laplacian variance high-freq energy proxy (higher = sharper)
        g = img.mean(1, keepdim=True); ker = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]], device=dev).view(1, 1, 3, 3)
        return float(F.conv2d(g, ker, padding=1).var().item())

    def save_mp4(path, frames, fps=4): imageio.mimwrite(path, frames, fps=fps, codec="libx264", quality=9, macro_block_size=1)
    def save_gif(path, frames, fps=4): imageio.mimwrite(path, frames, duration=1.0/fps, loop=0)
    def hcat(imgs, gap=6):
        h = imgs[0].shape[0]; sep = np.ones((h, gap, 3), np.uint8)*255
        out = [];
        for i, im in enumerate(imgs):
            out.append(im);
            if i < len(imgs)-1: out.append(sep)
        return np.concatenate(out, axis=1)
    def up2(img):  # 2x Lanczos upscale for crisper video display (resolution only; no new content/frames)
        return np.asarray(Image.fromarray(img).resize((img.shape[1]*2, img.shape[0]*2), Image.LANCZOS))

    # ---- chosen bake-off WINNER technique -> decoded u8 frames + honest protocol label (set after M3 gate) ----
    def winner_frames(fi, sv, nseed):
        zt = torch.tensor(lat[fi:fi+1], device=dev)
        if winner.startswith("sdedit_"): return [to_u8(f) for f in grid_frames(sdedit_grid(fi, sv, nseed, float(winner.split("_")[1]), 1.0), zt)]
        if winner.startswith("interp_"): return [to_u8(f) for f in grid_frames(interp_grid(fi, sv, nseed, float(winner.split("_")[1]), 1.0), zt)]
        if winner == "ar":            return [to_u8(f) for f in ar_diffusion_frames(fi, sv, nseed, False, 1.0)]
        if winner == "ar_reproj":     return [to_u8(f) for f in ar_diffusion_frames(fi, sv, nseed, True, 1.0)]
        if winner.startswith("direct_g"): return direct_future(fi, sv, float(winner.split("_g")[1]))
        if winner == "direct":        return direct_future(fi, sv)
        return diff_future(fi, sv, nseed)  # fallback: single-pass diffusion
    WINNER_LABEL = {"sdedit_0.3": "DiT-WAM (regression-guided)", "sdedit_0.5": "DiT-WAM (regression-guided)",
                    "interp_0.5": "DiT-WAM (interpolated)", "ar": "DiT-WAM (open-loop chained)",
                    "ar_reproj": "DiT-WAM (open-loop, reproject)", "direct_g1.3": "DiT-direct (amplified)",
                    "direct": "DiT-direct", "diffusion": "DiT-diffusion"}.get(winner, f"DiT-WAM ({winner})")

    which = [w.strip() for w in which.split(",") if w.strip()]
    made = []

    if "f5" in which or "teaser" in which:
        with PdfPages(f"{OUT}/f5_multiscene.pdf") as pp:  # -> fig_vae_qualitative.pdf at report-figure step (M6)
            disp = [0, 3, 6, 9, 12, 15]
            for fi in picks:
                zt = torch.tensor(lat[fi:fi+1], device=dev)
                gt = [to_u8(decode(torch.tensor(lat[fi+1+k:fi+2+k], device=dev))) for k in disp]
                ddf = direct_future(fi, steers[fi]); dd = [ddf[k] for k in disp]
                ggf = diff_future(fi, steers[fi], seed); gg = [ggf[k] for k in disp]
                rr = [rgb_real(fi+1+k) if rgb_real(fi+1+k) is not None else gt[i] for i, k in enumerate(disp)]
                rows = [("Camera (RGB)", rr), ("VAE-GT (ceiling)", gt),
                        ("DiT-direct (motion, blurry)", dd), ("DiT-diffusion (sharp, single-pass)", gg)]
                if winner:  # 5th row = chosen moving-row WINNER (DiT-WAM); empty winner keeps the 4-row honest fallback
                    wwf = winner_frames(fi, steers[fi], seed); rows.append((WINNER_LABEL, [wwf[k] for k in disp]))
                fig, ax = plt.subplots(len(rows), len(disp), figsize=(2.4*len(disp), 2.4*len(rows)))
                for r, (lab, ims) in enumerate(rows):
                    for c, k in enumerate(disp):
                        ax[r, c].imshow(ims[c]); ax[r, c].axis("off")
                        if r == 0: ax[r, c].set_title(f"t+{k}", fontsize=10)
                        if c == 0: ax[r, c].text(-0.08, 0.5, lab, rotation=90, va="center", ha="right", transform=ax[r, c].transAxes, fontsize=8)
                fig.tight_layout(); pp.savefig(fig, dpi=120); plt.close(fig)
                if "teaser" in which:  # teaser components @ t+15
                    imageio.imwrite(f"{OUT}/teaser_s{fi}_present.png", rr[0])
                    imageio.imwrite(f"{OUT}/teaser_s{fi}_direct.png", dd[-1])
                    imageio.imwrite(f"{OUT}/teaser_s{fi}_diffusion.png", gg[-1])
                    if winner: imageio.imwrite(f"{OUT}/teaser_s{fi}_winner.png", winner_frames(fi, steers[fi], seed)[-1])
        made.append("f5_multiscene.pdf (5-row qualitative)")

    if "counter" in which:
        for fi in picks:
            L = diff_future(fi, p5, seed, cfg_w); Smid = diff_future(fi, pmid, seed, cfg_w); R = diff_future(fi, p95, seed, cfg_w)
            frames = [up2(hcat([L[k], Smid[k], R[k]])) for k in range(HORIZON)]
            save_mp4(f"{OUT}/v1_counterfactual_s{fi}.mp4", frames, fps=4)
            save_gif(f"{OUT}/v1_counterfactual_s{fi}.gif", frames, fps=4)
        made.append(f"v1_counterfactual (L|S|R, cfg_w={cfg_w})")

    if "steersweep" in which:
        sweep = np.linspace(p5, p95, 9)
        for fi in picks:
            for j, sv in enumerate(sweep):
                fr = diff_future(fi, float(sv), seed, cfg_w)[15]  # endpoint t+15, fixed noise -> only steer changes
                imageio.imwrite(f"{OUT}/steersweep_s{fi}_k{j}.png", fr)
        made.append("steersweep (9 fixed-noise steers/scene for widget)")

    if "rollout" in which:
        for fi in picks:
            dseq = direct_future(fi, steers[fi]); gseq = diff_future(fi, steers[fi], seed)
            frames = [up2(hcat([dseq[k], gseq[k]])) for k in range(HORIZON)]  # direct | diffusion over t+1..t+16
            save_mp4(f"{OUT}/v2_direct_vs_diffusion_s{fi}.mp4", frames, fps=3)
            save_gif(f"{OUT}/v2_direct_vs_diffusion_s{fi}.gif", frames, fps=3)
        made.append("v2_direct_vs_diffusion (sharp vs blur, 3fps)")

    if "motion" in which:  # V-motion: chosen WINNER technique over t+1..t+16 (the "scene prediction" video)
        for fi in picks:
            wf = winner_frames(fi, steers[fi], seed)
            frames = [up2(wf[k]) for k in range(HORIZON)]
            save_mp4(f"{OUT}/vmotion_s{fi}.mp4", frames, fps=4)
            save_gif(f"{OUT}/vmotion_s{fi}.gif", frames, fps=4)
        made.append(f"vmotion ({winner})")

    if "bakeoff" in which:  # M2: inference-only moving-row candidates + automatic motion gate (dir-corr/disp/sharp)
        import json
        cands = ["direct", "direct_g1.3", "diffusion", "sdedit_0.3", "sdedit_0.5", "interp_0.5", "ar", "ar_reproj"]
        boj = {"blur_sigma": BLUR_SIGMA, "cfg_w": 1.0, "note": "metrics on mined scenes; dir_corr/disp vs GT, sharp=Laplacian-var", "scenes": []}
        for fi in picks:
            zt = torch.tensor(lat[fi:fi+1], device=dev)
            gtf = [decode(torch.tensor(lat[fi+1+k:fi+2+k], device=dev)) for k in range(HORIZON)]
            gtd = _disp_seq(gtf); gtlo, gthi = _seq_lowhigh(gtf)
            gen_map = {
                "direct":      lambda: grid_frames(direct_grid(fi, steers[fi], 1.0), zt),
                "direct_g1.3": lambda: grid_frames(direct_grid(fi, steers[fi], 1.3), zt),
                "diffusion":   lambda: grid_frames(diff_grid(fi, steers[fi], seed, 1.0), zt),
                "sdedit_0.3":  lambda: grid_frames(sdedit_grid(fi, steers[fi], seed, 0.3, 1.0), zt),
                "sdedit_0.5":  lambda: grid_frames(sdedit_grid(fi, steers[fi], seed, 0.5, 1.0), zt),
                "interp_0.5":  lambda: grid_frames(interp_grid(fi, steers[fi], seed, 0.5, 1.0), zt),
                "ar":          lambda: ar_diffusion_frames(fi, steers[fi], seed, False, 1.0),
                "ar_reproj":   lambda: ar_diffusion_frames(fi, steers[fi], seed, True, 1.0),
            }
            srec = {"fi": int(fi), "gt": {"lo": round(gtlo, 4), "hi": round(gthi, 4), "disp_mag": round(_disp_mag(gtd), 3)}}
            for name in cands:
                fr = gen_map[name]()
                lo, hi = _seq_lowhigh(fr); ds = _disp_seq(fr)
                srec[name] = {"dir_corr": round(_dir_corr(ds, gtd), 3),
                              "disp_mag_frac": round(_disp_mag(ds) / (_disp_mag(gtd) + 1e-9), 3),
                              "lo_frac": round(lo / (gtlo + 1e-9), 3), "hi_frac": round(hi / (gthi + 1e-9), 3),
                              "sharp": round(_sharp(fr[HORIZON // 2]), 5)}
                imageio.imwrite(f"{OUT}/bakeoff_s{fi}_{name}.png", hcat([to_u8(fr[k]) for k in [0, 3, 6, 9, 12, 15]]))
            boj["scenes"].append(srec)
            print(f"[bakeoff] fi{fi} (gt disp {srec['gt']['disp_mag']}): " +
                  " | ".join(f"{n}:dc{srec[n]['dir_corr']},dm{srec[n]['disp_mag_frac']},sh{srec[n]['sharp']}" for n in cands))
        with open(f"{OUT}/motion_viz_bakeoff.json", "w") as f: json.dump(boj, f, indent=2)
        made.append(f"bakeoff ({len(picks)}sc x {len(cands)}cand + filmstrips)")

    vol.commit()
    print("[viz] rendered:", made)
    print("[viz] outputs in", OUT, "->", sorted(os.listdir(OUT))[:30])
    return {"made": made, "out": OUT}


def _entry(fn):
    return app.local_entrypoint()(fn) if app is not None else fn


@_entry
def main(which: str = "counter,steersweep,f5,teaser,rollout", n_scenes: int = 3, seed: int = 0, scene_ids: str = "",
         cfg_w: float = 1.0, winner: str = "sdedit_0.5"):
    print(render.remote(which, n_scenes, seed, scene_ids, cfg_w, winner))
