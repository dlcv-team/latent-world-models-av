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
def render(which: str = "counter,steersweep,f5,teaser,rollout", n_scenes: int = 3, seed: int = 0):
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

    def diff_future(fi, sv, nseed):  # 16 decoded diffusion frames for constant steer sv, fixed noise nseed
        zt = torch.tensor(lat[fi:fi+1], device=dev); ztn = (patchify(zt)-gzm)/gzs; a = g_fo(act_const(fi, sv))
        ns = 50; st = max(DIFFUSION_STEPS//ns, 1); ts = list(reversed(list(range(0, DIFFUSION_STEPS, st))[:ns]))
        gen = torch.Generator(device=dev).manual_seed(nseed); x = torch.randn(1, HORIZON*N_SPATIAL, PATCH_DIM, device=dev, generator=gen)
        for i, tv in enumerate(ts):
            t = torch.full((1,), tv, device=dev, dtype=torch.long); px0 = g_dit(x, ztn, a, t)
            at = alphas[tv]; ap = alphas[ts[i+1]] if i < len(ts)-1 else torch.tensor(1.0, device=dev)
            nd = (x - torch.sqrt(at)*px0)/torch.sqrt(1-at+1e-8); x = torch.sqrt(ap)*px0 + torch.sqrt(1-ap)*nd
        gp = (x*gzs+gzm).reshape(1, HORIZON, N_SPATIAL, PATCH_DIM)
        return [to_u8(decode(calib(unpatchify(gp[:, k]), zt))) for k in range(HORIZON)]
    def direct_future(fi, sv):
        zt = torch.tensor(lat[fi:fi+1], device=dev); ztn = (patchify(zt)-dzm)/dzs
        zr = ztn.unsqueeze(1).expand(-1, HORIZON, -1, -1).reshape(1, HORIZON*N_SPATIAL, PATCH_DIM)
        dp = (d_dit(zr, ztn, d_fo(act_const(fi, sv)), torch.zeros(1, dtype=torch.long, device=dev))*dzs+dzm).reshape(1, HORIZON, N_SPATIAL, PATCH_DIM)
        return [to_u8(decode(unpatchify(dp[:, k]))) for k in range(HORIZON)]

    def save_mp4(path, frames, fps=4): imageio.mimwrite(path, frames, fps=fps, codec="libx264", quality=8, macro_block_size=1)
    def save_gif(path, frames, fps=4): imageio.mimwrite(path, frames, duration=1.0/fps, loop=0)
    def hcat(imgs, gap=6):
        h = imgs[0].shape[0]; sep = np.ones((h, gap, 3), np.uint8)*255
        out = [];
        for i, im in enumerate(imgs):
            out.append(im);
            if i < len(imgs)-1: out.append(sep)
        return np.concatenate(out, axis=1)

    which = [w.strip() for w in which.split(",") if w.strip()]
    made = []

    if "f5" in which or "teaser" in which:
        with PdfPages(f"{OUT}/f5_multiscene.pdf") as pp:
            disp = [0, 4, 8, 12, 15]
            for fi in picks:
                zt = torch.tensor(lat[fi:fi+1], device=dev)
                gt = [to_u8(decode(torch.tensor(lat[fi+1+k:fi+2+k], device=dev))) for k in disp]
                dd = [direct_future(fi, steers[fi])[k] for k in disp]
                gg_full = diff_future(fi, steers[fi], seed); gg = [gg_full[k] for k in disp]
                rr = [rgb_real(fi+1+k) if rgb_real(fi+1+k) is not None else gt[i] for i, k in enumerate(disp)]
                fig, ax = plt.subplots(4, len(disp), figsize=(2.4*len(disp), 9.6))
                rows = [("Camera (RGB)", rr), ("VAE-GT", gt), ("DiT-direct (regression)", dd), ("DiT-diffusion", gg)]
                for r, (lab, ims) in enumerate(rows):
                    for c, k in enumerate(disp):
                        ax[r, c].imshow(ims[c]); ax[r, c].axis("off")
                        if r == 0: ax[r, c].set_title(f"t+{k}", fontsize=10)
                        if c == 0: ax[r, c].text(-0.08, 0.5, lab, rotation=90, va="center", ha="right", transform=ax[r, c].transAxes, fontsize=9)
                fig.tight_layout(); pp.savefig(fig, dpi=120); plt.close(fig)
                if "teaser" in which:  # teaser components: present, direct-blur, diffusion-sharp @ t+15
                    imageio.imwrite(f"{OUT}/teaser_s{fi}_present.png", rr[0])
                    imageio.imwrite(f"{OUT}/teaser_s{fi}_direct.png", dd[-1])
                    imageio.imwrite(f"{OUT}/teaser_s{fi}_diffusion.png", gg[-1])
        made.append("f5_multiscene.pdf")

    if "counter" in which:
        for fi in picks:
            L = diff_future(fi, p5, seed); Smid = diff_future(fi, pmid, seed); R = diff_future(fi, p95, seed)
            frames = [hcat([L[k], Smid[k], R[k]]) for k in range(HORIZON)]
            save_mp4(f"{OUT}/v1_counterfactual_s{fi}.mp4", frames, fps=4)
            save_gif(f"{OUT}/v1_counterfactual_s{fi}.gif", frames, fps=4)
        made.append("v1_counterfactual (L|S|R, fixed noise)")

    if "steersweep" in which:
        sweep = np.linspace(p5, p95, 9)
        for fi in picks:
            for j, sv in enumerate(sweep):
                fr = diff_future(fi, float(sv), seed)[15]  # endpoint t+15, fixed noise -> only steer changes
                imageio.imwrite(f"{OUT}/steersweep_s{fi}_k{j}.png", fr)
        made.append("steersweep (9 fixed-noise steers/scene for widget)")

    if "rollout" in which:
        for fi in picks:
            dseq = direct_future(fi, steers[fi]); gseq = diff_future(fi, steers[fi], seed)
            frames = [hcat([dseq[k], gseq[k]]) for k in range(HORIZON)]  # direct | diffusion over t+1..t+16
            save_mp4(f"{OUT}/v2_direct_vs_diffusion_s{fi}.mp4", frames, fps=3)
            save_gif(f"{OUT}/v2_direct_vs_diffusion_s{fi}.gif", frames, fps=3)
        made.append("v2_direct_vs_diffusion (sharp vs blur, 3fps)")

    vol.commit()
    print("[viz] rendered:", made)
    print("[viz] outputs in", OUT, "->", sorted(os.listdir(OUT))[:30])
    return {"made": made, "out": OUT}


def _entry(fn):
    return app.local_entrypoint()(fn) if app is not None else fn


@_entry
def main(which: str = "counter,steersweep,f5,teaser,rollout", n_scenes: int = 3):
    print(render.remote(which, n_scenes))
