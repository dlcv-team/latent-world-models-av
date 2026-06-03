"""Generative evaluation of VAE-latent DiT world model.

Evaluates DIRECT vs DIFFUSION (action-conditioned, optional classifier-free
guidance) predictions of future SD-VAE latents on a LOCKED deterministic window
set (first window of each test scene). Reports three axes so a generative win
cannot be "sharp but wrong":

  Axis 1 Realism      : Laplacian sharpness (calibrated as fraction of VAE-GT).
  Axis 2 Manifold     : min distance pred-latent -> nearest HELD-OUT TRAIN latent
                        (+ kNN-rank, to flag memorization).
  Axis 3 Fidelity GV4 : LPIPS + SSIM of decoded pred vs decoded VAE-GT (blur-
                        sensitive, unlike latent CosSim) -> "is it the RIGHT future".
  + diversity (best/mean-of-K vs a real-future band), latent CosSim (tradeoff),
    and an action-use check (true vs +0.3-steer vs shuffled, fixed noise).

Decode/scaling identical to eval_vae_visual_modal.py (blur-causality control).

Usage:
  modal run scripts/eval_generative_vae_modal.py --models direct,diffusion --k 8 --n-windows 48
  modal run scripts/eval_generative_vae_modal.py --models expb --k 8 --n-windows 32   # B0 preview
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

try:
    import modal
except ImportError:
    modal = None

SCRIPT_DIR = Path(__file__).resolve().parent
TRAIN_SCRIPT = SCRIPT_DIR / "train_dit_vae_modal.py"

if modal is not None:
    app = modal.App("lwm-av-gen-eval")
    vol = modal.Volume.from_name("nuscenes-full")
    image = (
        modal.Image.debian_slim(python_version="3.12")
        .pip_install(
            "torch==2.5.1", "numpy>=1.26", "Pillow>=10.0", "diffusers>=0.27",
            "matplotlib>=3.8", "accelerate", "transformers>=4.50", "torchvision>=0.20",
            "lpips>=0.1.4", "torchmetrics>=1.2", "scipy>=1.11", "torch-fidelity>=0.3.0",
        )
        .add_local_file(str(TRAIN_SCRIPT), remote_path="/root/train_dit_vae_modal.py")
    )
else:
    app = None
    vol = None
    image = None

VOL_PATH = "/vol"
SPATIAL_DIR = f"{VOL_PATH}/embeddings/spatial"
CKPT_DIR = f"{VOL_PATH}/dits/vae_latent"
VAE_NPZ = f"{SPATIAL_DIR}/sd_vae_latents.npz"
OUT_DIR = f"{SPATIAL_DIR}/gen_eval"
SCALING = 0.18215
HORIZON = 16
DIFFUSION_STEPS = 1000

CKPT_PATHS = {
    "direct": f"{CKPT_DIR}/h{HORIZON}/seed_0/dit.pt",
    "diffusion": f"{CKPT_DIR}/diffusion/h{HORIZON}/seed_0/dit.pt",
    "expb": f"{CKPT_DIR}/diffusion_ad0.3/h{HORIZON}/seed_0/dit.pt",
}


def _decorator(fn):
    if app is not None:
        return app.function(volumes={VOL_PATH: vol}, image=image, gpu="A10G",
                            timeout=7200, memory=32768)(fn)
    return fn


@_decorator
def gen_eval(models: str = "diffusion", k: int = 8, cfg_weights: str = "1.0",
            n_windows: int = 48, steps_eval: str = "3,15", n_fig: int = 5, seed: int = 0):
    import numpy as np
    import torch
    import torch.nn.functional as F
    import lpips as lpips_lib
    from torchmetrics.functional import structural_similarity_index_measure as ssim_fn
    from diffusers import AutoencoderKL

    spec = importlib.util.spec_from_file_location("tv", "/root/train_dit_vae_modal.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    patchify, unpatchify = mod.patchify, mod.unpatchify
    AnchoredVAEDiT, FourierActionEmbedding = mod.AnchoredVAEDiT, mod.FourierActionEmbedding
    DIT_CONFIG, FOURIER_CONFIG = mod.DIT_CONFIG, mod.FOURIER_CONFIG
    PATCH_DIM, N_SPATIAL, GRID_H, GRID_W = mod.PATCH_DIM, mod.N_SPATIAL, mod.GRID_H, mod.GRID_W
    CosineNoiseSchedule = mod._define_noise_schedule()

    device = torch.device("cuda")
    torch.manual_seed(seed); np.random.seed(seed)
    model_list = [m.strip() for m in models.split(",") if m.strip()]
    cfg_list = [float(w) for w in cfg_weights.split(",") if w.strip()]
    steps = [int(s) for s in steps_eval.split(",") if s.strip()]
    os.makedirs(OUT_DIR, exist_ok=True)

    schedule = CosineNoiseSchedule(n_steps=DIFFUSION_STEPS).to(device)
    alphas = schedule.alphas_cumprod
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device).eval()
    lpips_net = lpips_lib.LPIPS(net="alex").to(device).eval()

    data = np.load(VAE_NPZ, allow_pickle=True)
    latents = data["vae_latents"].astype(np.float32)   # (N,4,32,32), scaled
    scenes = data["scene_names"]; splits = data["splits"]
    steers = data["steer_norms"].astype(np.float32); accels = data["accel_norms"].astype(np.float32)

    # ---- locked deterministic eval windows: first window of each test scene ----
    test_idx = np.where(splits == "test")[0]
    picks = []
    for sc in np.unique(scenes[test_idx]):
        idx = test_idx[scenes[test_idx] == sc]
        if len(idx) > HORIZON:
            picks.append(int(idx[0]))
    picks = picks[:n_windows]
    n_win = len(picks)
    print(f"[gen-eval] locked windows: {n_win}; models={model_list}; k={k}; cfg={cfg_list}; steps={steps}")

    # ---- held-out TRAIN latent bank for manifold adherence (patch-token level) ----
    train_idx = np.where(splits == "train")[0]
    rng = np.random.default_rng(0)
    bank_sel = rng.choice(train_idx, size=min(4000, len(train_idx)), replace=False)
    bank = torch.tensor(latents[bank_sel], device=device)               # (Tb,4,32,32)
    bank_grid = bank.reshape(bank.shape[0], -1)                          # (Tb, 4096) full-grid (primary)
    bank_tok = patchify(bank).reshape(-1, PATCH_DIM)                     # (Tb*64, 64)  token-level (2ndary)
    bank_tok = bank_tok[rng.choice(bank_tok.shape[0], size=min(60000, bank_tok.shape[0]), replace=False)]
    print(f"[gen-eval] manifold bank: grid={bank_grid.shape} tokens={bank_tok.shape}")

    # z-normalization from any ckpt (all share train stats); load lazily per model
    def load_model(ckpt_path, use_ema=True):
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        dit = AnchoredVAEDiT(horizon=HORIZON, n_spatial=N_SPATIAL, **DIT_CONFIG).to(device)
        fou = FourierActionEmbedding(**FOURIER_CONFIG).to(device)
        dit.load_state_dict(ck["dit"]); fou.load_state_dict(ck["fourier"])
        if use_ema and "ema" in ck and ck["ema"]:
            ema = ck["ema"]
            for nm, p in dit.named_parameters():
                if nm in ema: p.data.copy_(ema[nm].to(device))
            for nm, p in fou.named_parameters():
                if nm in ema: p.data.copy_(ema[nm].to(device))
        dit.eval(); fou.eval()
        return dit, fou, ck["z_mean"].to(device), ck["z_std"].to(device)

    def gather(pick_list):
        z_t = torch.tensor(np.stack([latents[i] for i in pick_list]), device=device)      # (W,4,32,32)
        zf = torch.tensor(np.stack([latents[i + 1:i + 1 + HORIZON] for i in pick_list]), device=device)  # (W,H,4,32,32)
        act = torch.tensor(np.stack([
            np.stack([[steers[i + kk], accels[i + kk]] for kk in range(HORIZON)]) for i in pick_list
        ]), dtype=torch.float32, device=device)                                            # (W,H,2)
        return z_t, zf, act

    z_t_all, zf_all, act_all = gather(picks)

    def norm_grid(g, z_mean, z_std):                # (B,4,32,32) -> (B,64,64) normalized tokens
        return (patchify(g) - z_mean) / z_std

    def denorm(tok, z_mean, z_std):
        return tok * z_std + z_mean

    def decode(latent_grid):                        # (B,4,32,32) scaled -> (B,3,256,256) in [-1,1]
        return vae.decode(latent_grid.clamp(-6, 6) / SCALING).sample.clamp(-1, 1)

    def sharpness(img01):                           # (B,3,256,256) in [0,1] -> per-img laplacian var
        g = img01.mean(1, keepdim=True)
        ker = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32, device=device).view(1, 1, 3, 3)
        lap = F.conv2d(g, ker, padding=1)
        return lap.view(lap.shape[0], -1).var(dim=1)

    def to01(img):
        return (img + 1) / 2

    def manifold_dist(pred_grid):                   # (B,4,32,32) -> {grid, token} mean min-dist to real bank
        # GRID-level (4096-d, PRIMARY): high-dim conditional mean lands off-manifold;
        # a realistic sample lands near a real latent. This is the GV2 metric.
        pg = pred_grid.reshape(pred_grid.shape[0], -1)
        dg = torch.cdist(pg, bank_grid).min(dim=1).values.mean().item()
        # TOKEN-level (64-d, secondary): blurry mean is near common smooth tokens, so this
        # does NOT capture realism -- kept only for transparency (see B0 finding).
        tok = patchify(pred_grid).reshape(-1, PATCH_DIM)
        dt = torch.cdist(tok, bank_tok).min(dim=1).values.mean().item()
        return {"grid": round(dg, 4), "token": round(dt, 4)}

    def direct_predict(dit, fou, z_t_n):
        W = z_t_n.shape[0]
        z_rep = z_t_n.unsqueeze(1).expand(-1, HORIZON, -1, -1).reshape(W, HORIZON * N_SPATIAL, PATCH_DIM)
        t0 = torch.zeros(W, dtype=torch.long, device=device)
        return dit(z_rep, z_t_n, fou(act_seq_holder["a"]), t0)        # (W, H*S, Pd) normalized

    def ddim_cfg(dit, fou, z_t_n, a_cond, cfg_w, gen):
        W = z_t_n.shape[0]
        n_steps = 50
        stride = max(DIFFUSION_STEPS // n_steps, 1)
        ts = list(reversed(list(range(0, DIFFUSION_STEPS, stride))[:n_steps]))
        a_zero = torch.zeros_like(a_cond)
        x = torch.randn(W, HORIZON * N_SPATIAL, PATCH_DIM, device=device, generator=gen)
        for i, tv in enumerate(ts):
            t = torch.full((W,), tv, device=device, dtype=torch.long)
            px0 = dit(x, z_t_n, a_cond, t)
            if cfg_w != 1.0:
                px0_u = dit(x, z_t_n, a_zero, t)
                px0 = px0_u + cfg_w * (px0 - px0_u)
            at = alphas[tv]
            ap = alphas[ts[i + 1]] if i < len(ts) - 1 else torch.tensor(1.0, device=device)
            nd = (x - torch.sqrt(at) * px0) / torch.sqrt(1 - at + 1e-8)
            x = torch.sqrt(ap) * px0 + torch.sqrt(1 - ap) * nd
        return x

    act_seq_holder = {"a": act_all}
    results = {"n_windows": n_win, "k": k, "steps_eval": steps, "models": {}}

    # GT decode + sharpness (reference) at eval steps
    gt_sharp = {}
    gt_imgs = {}
    with torch.no_grad():
        for st in steps:
            gimg = decode(zf_all[:, st])
            gt_imgs[st] = gimg
            gt_sharp[st] = sharpness(to01(gimg)).mean().item()
    results["gt_sharpness"] = {str(s): round(gt_sharp[s], 2) for s in steps}

    def eval_pred_grids(pred_grids_by_step, label):
        """pred_grids_by_step: dict step-> (W,4,32,32). Compute realism/manifold/fidelity vs GT."""
        out = {"sharpness": {}, "sharp_frac_of_gt": {}, "manifold": {}, "lpips": {}, "ssim": {}, "latent_cos": {}}
        with torch.no_grad():
            for st in steps:
                pg = pred_grids_by_step[st]
                pimg = decode(pg)
                sh = sharpness(to01(pimg)).mean().item()
                out["sharpness"][str(st)] = round(sh, 2)
                out["sharp_frac_of_gt"][str(st)] = round(sh / (gt_sharp[st] + 1e-9), 3)
                out["manifold"][str(st)] = manifold_dist(pg)
                out["lpips"][str(st)] = round(lpips_net(pimg, gt_imgs[st]).mean().item(), 4)
                out["ssim"][str(st)] = round(ssim_fn(to01(pimg), to01(gt_imgs[st])).item(), 4)
                # latent cosine (per-token mean) on patch tokens
                pt = patchify(pg); gt_t = patchify(zf_all[:, st])
                out["latent_cos"][str(st)] = round(F.cosine_similarity(pt, gt_t, dim=-1).mean().item(), 4)
        print(f"  [{label}] sharp_frac_gt={out['sharp_frac_of_gt']} lpips={out['lpips']} "
              f"manifold={out['manifold']} latent_cos={out['latent_cos']}")
        return out

    for m in model_list:
        ckpt_path = CKPT_PATHS.get(m, m)
        if not os.path.exists(ckpt_path):
            print(f"[gen-eval] MISSING ckpt for {m}: {ckpt_path}; skipping")
            results["models"][m] = {"error": "ckpt_missing", "path": ckpt_path}
            continue
        dit, fou, z_mean, z_std = load_model(ckpt_path)
        z_t_n = norm_grid(z_t_all, z_mean, z_std)
        a_cond = fou(act_all)

        if m == "direct":
            with torch.no_grad():
                pred_tok = direct_predict(dit, fou, z_t_n)
                pred_tok = denorm(pred_tok, z_mean, z_std).reshape(n_win, HORIZON, N_SPATIAL, PATCH_DIM)
                grids = {st: unpatchify(pred_tok[:, st]) for st in steps}
            results["models"]["direct"] = eval_pred_grids(grids, "direct")
            results["models"]["direct"]["_grids_step"] = steps  # marker
            saved_direct_grids = grids
            continue

        # diffusion model: for each cfg weight, sample K, compute mean/best metrics + diversity
        results["models"][m] = {}
        for w in cfg_list:
            with torch.no_grad():
                samples = []  # list over k of dict step->(W,4,32,32)
                sample_tok = []  # for diversity
                for ki in range(k):
                    gen = torch.Generator(device=device).manual_seed(1000 + ki)
                    tok = ddim_cfg(dit, fou, z_t_n, a_cond, w, gen)
                    tok_d = denorm(tok, z_mean, z_std).reshape(n_win, HORIZON, N_SPATIAL, PATCH_DIM)
                    sample_tok.append(tok_d)
                    samples.append({st: unpatchify(tok_d[:, st]) for st in steps})
                # realism/manifold/fidelity on sample 0 (representative); best/mean latent-cos over all k
                rep = eval_pred_grids(samples[0], f"{m}_w{w}_rep")
                # latent cos best/mean-of-k vs GT
                bestcos, meancos = {}, {}
                for st in steps:
                    gt_t = patchify(zf_all[:, st])
                    cs = torch.stack([
                        F.cosine_similarity(patchify(samples[ki][st]), gt_t, dim=-1).mean(dim=-1)
                        for ki in range(k)
                    ], dim=0)  # (k, W)
                    bestcos[str(st)] = round(cs.max(dim=0).values.mean().item(), 4)
                    meancos[str(st)] = round(cs.mean().item(), 4)
                # diversity: mean pairwise L2 across k samples (normalized tokens), last step
                st_div = steps[-1]
                flat = torch.stack([sample_tok[ki][:, st_div].reshape(n_win, -1) for ki in range(k)], dim=0)  # (k,W,D)
                div = 0.0
                if k >= 2:
                    for wi in range(n_win):
                        fl = flat[:, wi]  # (k,D)
                        pw = torch.cdist(fl, fl)
                        div += pw[torch.triu(torch.ones(k, k, device=device), 1) == 1].mean().item()
                    div /= n_win
                rep["best_of_k_latent_cos"] = bestcos
                rep["mean_of_k_latent_cos"] = meancos
                rep["diversity_l2"] = round(div, 4)
                results["models"][m][f"cfg_{w}"] = rep
                if w == 1.0:
                    saved_diff_grids = samples[0]  # representative for figure

    # ---- real-future variability band (calibrates diversity) ----
    # pairwise distance among real futures z_{t+H} across windows sharing similar z_t (kNN in pooled latent)
    with torch.no_grad():
        pooled = patchify(z_t_all).mean(1)                  # (W, 64)
        pooled = pooled / (pooled.norm(dim=-1, keepdim=True) + 1e-8)
        sims = pooled @ pooled.T
        st_div = steps[-1]
        futflat = patchify(zf_all[:, st_div]).reshape(n_win, -1)  # (W, 64*64)
        band = []
        for wi in range(n_win):
            row = sims[wi].clone(); row[wi] = -2
            nn = torch.topk(row, min(8, n_win - 1)).indices
            band.append((futflat[nn] - futflat[wi]).norm(dim=-1).mean().item())
        results["real_future_band_l2"] = round(float(np.mean(band)), 4)
    print(f"[gen-eval] real-future band L2 (step {steps[-1]}): {results['real_future_band_l2']}")

    with open(f"{OUT_DIR}/metrics_{'_'.join(model_list)}.json", "w") as f:
        json.dump(results, f, indent=2)
    vol.commit()
    print(json.dumps(results, indent=2))
    return results


@_decorator
def make_figure(n_fig: int = 5, cfg_w: float = 1.0, steps_show: str = "0,4,8,12,15", seed: int = 0):
    """4-row figure: RGB / VAE-GT / DiT-direct (blur) / DiT-diffusion (sharp)."""
    import numpy as np
    import torch
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from diffusers import AutoencoderKL

    spec = importlib.util.spec_from_file_location("tv", "/root/train_dit_vae_modal.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    patchify, unpatchify = mod.patchify, mod.unpatchify
    AnchoredVAEDiT, FourierActionEmbedding = mod.AnchoredVAEDiT, mod.FourierActionEmbedding
    DIT_CONFIG, FOURIER_CONFIG = mod.DIT_CONFIG, mod.FOURIER_CONFIG
    PATCH_DIM, N_SPATIAL = mod.PATCH_DIM, mod.N_SPATIAL
    CosineNoiseSchedule = mod._define_noise_schedule()
    device = torch.device("cuda")
    disp = [int(s) for s in steps_show.split(",")]
    os.makedirs(OUT_DIR, exist_ok=True)
    schedule = CosineNoiseSchedule(n_steps=DIFFUSION_STEPS).to(device)
    alphas = schedule.alphas_cumprod
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device).eval()

    data = np.load(VAE_NPZ, allow_pickle=True)
    latents = data["vae_latents"].astype(np.float32)
    scenes, splits = data["scene_names"], data["splits"]
    steers, accels = data["steer_norms"].astype(np.float32), data["accel_norms"].astype(np.float32)
    image_paths = data["image_paths"] if "image_paths" in data else None
    test_idx = np.where(splits == "test")[0]
    picks = []
    for sc in np.unique(scenes[test_idx]):
        idx = test_idx[scenes[test_idx] == sc]
        if len(idx) > HORIZON:
            picks.append(int(idx[0]))
    picks = picks[:n_fig]

    def load(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        dit = AnchoredVAEDiT(horizon=HORIZON, n_spatial=N_SPATIAL, **DIT_CONFIG).to(device)
        fou = FourierActionEmbedding(**FOURIER_CONFIG).to(device)
        dit.load_state_dict(ck["dit"]); fou.load_state_dict(ck["fourier"])
        if "ema" in ck and ck["ema"]:
            for nm, p in dit.named_parameters():
                if nm in ck["ema"]: p.data.copy_(ck["ema"][nm].to(device))
            for nm, p in fou.named_parameters():
                if nm in ck["ema"]: p.data.copy_(ck["ema"][nm].to(device))
        dit.eval(); fou.eval()
        return dit, fou, ck["z_mean"].to(device), ck["z_std"].to(device)

    d_dit, d_fou, dzm, dzs = load(CKPT_PATHS["direct"])
    g_dit, g_fou, gzm, gzs = load(CKPT_PATHS["diffusion"])

    def decode(grid):
        return vae.decode(grid.clamp(-6, 6) / SCALING).sample.clamp(-1, 1)

    def to_img(t):
        return ((t + 1) / 2)[0].permute(1, 2, 0).detach().cpu().numpy()

    pdf = f"{OUT_DIR}/vae_4row_demo.pdf"
    with PdfPages(pdf) as pp:
        for wi, fi in enumerate(picks):
            z_t = torch.tensor(latents[fi:fi + 1], device=device)
            act = torch.stack([torch.tensor([steers[fi + kk], accels[fi + kk]], device=device)
                               for kk in range(HORIZON)]).unsqueeze(0)
            zf = torch.tensor(latents[fi + 1:fi + 1 + HORIZON], device=device).unsqueeze(0)
            with torch.no_grad():
                zt_n = (patchify(z_t) - dzm) / dzs
                zr = zt_n.unsqueeze(1).expand(-1, HORIZON, -1, -1).reshape(1, HORIZON * N_SPATIAL, PATCH_DIM)
                t0 = torch.zeros(1, dtype=torch.long, device=device)
                dpred = (d_dit(zr, zt_n, d_fou(act), t0) * dzs + dzm).reshape(1, HORIZON, N_SPATIAL, PATCH_DIM)
                ztn_g = (patchify(z_t) - gzm) / gzs
                a_c = g_fou(act); a_z = torch.zeros_like(a_c)
                n_steps = 50; stride = max(DIFFUSION_STEPS // n_steps, 1)
                ts = list(reversed(list(range(0, DIFFUSION_STEPS, stride))[:n_steps]))
                gen = torch.Generator(device=device).manual_seed(seed)
                x = torch.randn(1, HORIZON * N_SPATIAL, PATCH_DIM, device=device, generator=gen)
                for i, tv in enumerate(ts):
                    t = torch.full((1,), tv, device=device, dtype=torch.long)
                    px0 = g_dit(x, ztn_g, a_c, t)
                    if cfg_w != 1.0:
                        pu = g_dit(x, ztn_g, a_z, t); px0 = pu + cfg_w * (px0 - pu)
                    at = alphas[tv]; ap = alphas[ts[i + 1]] if i < len(ts) - 1 else torch.tensor(1.0, device=device)
                    nd = (x - torch.sqrt(at) * px0) / torch.sqrt(1 - at + 1e-8)
                    x = torch.sqrt(ap) * px0 + torch.sqrt(1 - ap) * nd
                gpred = (x * gzs + gzm).reshape(1, HORIZON, N_SPATIAL, PATCH_DIM)

            fig, ax = plt.subplots(4, len(disp), figsize=(3 * len(disp), 12))
            for col, k in enumerate(disp):
                gt_dec = to_img(decode(zf[:, k]))
                rgb = gt_dec
                if image_paths is not None and fi + 1 + k < len(image_paths):
                    ip = f"{VOL_PATH}/nuscenes/{image_paths[fi + 1 + k]}"
                    if os.path.exists(ip):
                        from PIL import Image
                        import torchvision.transforms.functional as TF
                        im = Image.open(ip).convert("RGB"); w, h = im.size
                        c = min(w, h)
                        im = im.crop(((w - c) // 2, (h - c) // 2, (w - c) // 2 + c, (h - c) // 2 + c)).resize((256, 256))
                        rgb = TF.to_tensor(im).permute(1, 2, 0).numpy()
                dd = to_img(decode(unpatchify(dpred[:, k])))
                # diffusion: per-channel mean-match to the INPUT frame z_t channel stats
                # (visualization calibration; corrects the systematic latent channel offset / tint)
                gp_grid = unpatchify(gpred[:, k])
                zt_ch = z_t.mean(dim=(2, 3)).view(1, 4, 1, 1)
                gp_grid = gp_grid - gp_grid.mean(dim=(2, 3)).view(1, 4, 1, 1) + zt_ch
                gd = to_img(decode(gp_grid))
                for r, (img, title) in enumerate([(rgb, f"RGB t+{k}"), (gt_dec, f"VAE-GT t+{k}"),
                                                   (dd, f"DiT-direct t+{k}"), (gd, f"DiT-diffusion(cal) t+{k}")]):
                    ax[r, col].imshow(np.clip(img, 0, 1)); ax[r, col].set_title(title, fontsize=8); ax[r, col].axis("off")
            fig.tight_layout(); pp.savefig(fig); plt.close(fig)
    vol.commit()
    print(f"[figure] saved {pdf} ({len(picks)} windows, cfg_w={cfg_w})")
    return {"pdf": pdf, "n": len(picks), "cfg_w": cfg_w}


@_decorator
def action_use(n_windows: int = 48, perturb: float = 0.3, seed: int = 0):
    """Does the VAE diffusion model USE actions? Same x_T; true vs +perturb-steer vs
    time-shuffled action sequences. perturb_sens > shuffle_sens => uses structured actions."""
    import numpy as np
    import torch
    import torch.nn.functional as F

    spec = importlib.util.spec_from_file_location("tv", "/root/train_dit_vae_modal.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    patchify = mod.patchify
    AnchoredVAEDiT, FourierActionEmbedding = mod.AnchoredVAEDiT, mod.FourierActionEmbedding
    DIT_CONFIG, FOURIER_CONFIG = mod.DIT_CONFIG, mod.FOURIER_CONFIG
    PATCH_DIM, N_SPATIAL = mod.PATCH_DIM, mod.N_SPATIAL
    CosineNoiseSchedule = mod._define_noise_schedule()
    device = torch.device("cuda")
    schedule = CosineNoiseSchedule(n_steps=DIFFUSION_STEPS).to(device)
    alphas = schedule.alphas_cumprod

    data = np.load(VAE_NPZ, allow_pickle=True)
    latents = data["vae_latents"].astype(np.float32)
    scenes, splits = data["scene_names"], data["splits"]
    steers, accels = data["steer_norms"].astype(np.float32), data["accel_norms"].astype(np.float32)
    test_idx = np.where(splits == "test")[0]
    picks = []
    for sc in np.unique(scenes[test_idx]):
        idx = test_idx[scenes[test_idx] == sc]
        if len(idx) > HORIZON:
            picks.append(int(idx[0]))
    picks = picks[:n_windows]
    W = len(picks)

    ck = torch.load(CKPT_PATHS["diffusion"], map_location=device, weights_only=False)
    dit = AnchoredVAEDiT(horizon=HORIZON, n_spatial=N_SPATIAL, **DIT_CONFIG).to(device)
    fou = FourierActionEmbedding(**FOURIER_CONFIG).to(device)
    dit.load_state_dict(ck["dit"]); fou.load_state_dict(ck["fourier"])
    if "ema" in ck and ck["ema"]:
        for nm, p in dit.named_parameters():
            if nm in ck["ema"]: p.data.copy_(ck["ema"][nm].to(device))
        for nm, p in fou.named_parameters():
            if nm in ck["ema"]: p.data.copy_(ck["ema"][nm].to(device))
    dit.eval(); fou.eval()
    zm, zs = ck["z_mean"].to(device), ck["z_std"].to(device)

    z_t = torch.tensor(np.stack([latents[i] for i in picks]), device=device)
    act = torch.tensor(np.stack([
        np.stack([[steers[i + kk], accels[i + kk]] for kk in range(HORIZON)]) for i in picks
    ]), dtype=torch.float32, device=device)                       # (W,H,2)
    act_pert = act.clone(); act_pert[:, :, 0] = (act_pert[:, :, 0] + perturb).clamp(-1, 1)
    g = torch.Generator(device=device).manual_seed(123)
    perm = torch.stack([torch.randperm(HORIZON, generator=g, device=device) for _ in range(W)])
    act_shuf = torch.stack([act[w, perm[w]] for w in range(W)])

    ztn = (patchify(z_t) - zm) / zs

    def sample(a_seq, gseed):
        a = fou(a_seq)
        n_steps = 50; stride = max(DIFFUSION_STEPS // n_steps, 1)
        ts = list(reversed(list(range(0, DIFFUSION_STEPS, stride))[:n_steps]))
        gen = torch.Generator(device=device).manual_seed(gseed)
        x = torch.randn(W, HORIZON * N_SPATIAL, PATCH_DIM, device=device, generator=gen)
        for i, tv in enumerate(ts):
            t = torch.full((W,), tv, device=device, dtype=torch.long)
            px0 = dit(x, ztn, a, t)
            at = alphas[tv]; ap = alphas[ts[i + 1]] if i < len(ts) - 1 else torch.tensor(1.0, device=device)
            nd = (x - torch.sqrt(at) * px0) / torch.sqrt(1 - at + 1e-8)
            x = torch.sqrt(ap) * px0 + torch.sqrt(1 - ap) * nd
        return x.reshape(W, HORIZON, N_SPATIAL, PATCH_DIM)

    with torch.no_grad():
        base = sample(act, 7)            # same x_T (gseed) across conditions
        pert = sample(act_pert, 7)
        shuf = sample(act_shuf, 7)
        out = {"perturb": perturb, "n_windows": W, "perturb_sens_by_step": {}, "shuffle_sens_by_step": {}}
        for st in [3, 15]:
            ps = (1 - F.cosine_similarity(base[:, st], pert[:, st], dim=-1)).mean().item()
            ss = (1 - F.cosine_similarity(base[:, st], shuf[:, st], dim=-1)).mean().item()
            out["perturb_sens_by_step"][str(st)] = round(ps, 4)
            out["shuffle_sens_by_step"][str(st)] = round(ss, 4)
    out["uses_actions"] = out["perturb_sens_by_step"]["15"] > 0.01
    print(f"[action-use] perturb_sens={out['perturb_sens_by_step']} shuffle_sens={out['shuffle_sens_by_step']}")
    return out


@_decorator
def fid_eval(n_windows: int = 300, horizons: str = "3", cfg_w: float = 1.0, seed: int = 0,
            diffusion_ckpt: str = "diffusion", windows_per_scene: int = 1):
    """FID + KID of decoded {direct, diffusion(raw+channel-calibrated), VAE-GT} vs real test RGB.
    Held-out TEST scenes only. Records the frozen window manifest. Per-channel latent
    diagnostic (diffusion vs real) drives the channel-calibration (= visualization calibration)."""
    import numpy as np
    import torch
    import torch.nn.functional as F
    from diffusers import AutoencoderKL
    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchmetrics.image.kid import KernelInceptionDistance
    from PIL import Image
    import torchvision.transforms.functional as TF

    spec = importlib.util.spec_from_file_location("tv", "/root/train_dit_vae_modal.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    patchify, unpatchify = mod.patchify, mod.unpatchify
    AnchoredVAEDiT, FourierActionEmbedding = mod.AnchoredVAEDiT, mod.FourierActionEmbedding
    DIT_CONFIG, FOURIER_CONFIG = mod.DIT_CONFIG, mod.FOURIER_CONFIG
    PATCH_DIM, N_SPATIAL = mod.PATCH_DIM, mod.N_SPATIAL
    CosineNoiseSchedule = mod._define_noise_schedule()
    device = torch.device("cuda")
    hs = [int(x) for x in horizons.split(",")]
    schedule = CosineNoiseSchedule(n_steps=DIFFUSION_STEPS).to(device)
    alphas = schedule.alphas_cumprod
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device).eval()
    os.makedirs(OUT_DIR, exist_ok=True)

    data = np.load(VAE_NPZ, allow_pickle=True)
    latents = data["vae_latents"].astype(np.float32)
    scenes, splits = data["scene_names"], data["splits"]
    steers, accels = data["steer_norms"].astype(np.float32), data["accel_norms"].astype(np.float32)
    image_paths = data["image_paths"] if "image_paths" in data else None
    test_idx = np.where(splits == "test")[0]

    # frozen manifest: up to windows_per_scene per held-out test scene, RGB must exist
    picks = []
    for sc in np.unique(scenes[test_idx]):
        idx = test_idx[scenes[test_idx] == sc]
        cnt = 0
        for j in range(len(idx) - HORIZON):
            fi = int(idx[j])
            ok = image_paths is not None and all(
                fi + 1 + h < len(image_paths) and
                os.path.exists(f"{VOL_PATH}/nuscenes/{image_paths[fi + 1 + h]}") for h in hs)
            if ok:
                picks.append(fi); cnt += 1
                if cnt >= windows_per_scene:
                    break
        if len(picks) >= n_windows:
            break
    picks = picks[:n_windows]
    print(f"[fid] frozen manifest: {len(picks)} windows (held-out test), horizons={hs}")

    def load(tag):
        ck = torch.load(CKPT_PATHS.get(tag, tag), map_location=device, weights_only=False)
        dit = AnchoredVAEDiT(horizon=HORIZON, n_spatial=N_SPATIAL, **DIT_CONFIG).to(device)
        fou = FourierActionEmbedding(**FOURIER_CONFIG).to(device)
        dit.load_state_dict(ck["dit"]); fou.load_state_dict(ck["fourier"])
        if "ema" in ck and ck["ema"]:
            for nm, p in dit.named_parameters():
                if nm in ck["ema"]: p.data.copy_(ck["ema"][nm].to(device))
            for nm, p in fou.named_parameters():
                if nm in ck["ema"]: p.data.copy_(ck["ema"][nm].to(device))
        dit.eval(); fou.eval()
        return dit, fou, ck["z_mean"].to(device), ck["z_std"].to(device)

    d_dit, d_fou, dzm, dzs = load("direct")
    g_dit, g_fou, gzm, gzs = load(diffusion_ckpt)

    def decode(grid):
        return ((vae.decode(grid.clamp(-6, 6) / SCALING).sample.clamp(-1, 1)) + 1) / 2  # ->[0,1]

    def load_rgb(fi, h):
        im = Image.open(f"{VOL_PATH}/nuscenes/{image_paths[fi + 1 + h]}").convert("RGB")
        w, hh = im.size; c = min(w, hh)
        im = im.crop(((w - c) // 2, (hh - c) // 2, (w - c) // 2 + c, (hh - c) // 2 + c)).resize((256, 256))
        return TF.to_tensor(im).unsqueeze(0).to(device)

    def direct_pred(z_t):
        ztn = (patchify(z_t) - dzm) / dzs
        zr = ztn.unsqueeze(1).expand(-1, HORIZON, -1, -1).reshape(z_t.shape[0], HORIZON * N_SPATIAL, PATCH_DIM)
        t0 = torch.zeros(z_t.shape[0], dtype=torch.long, device=device)
        return (d_dit(zr, ztn, d_fou(act_h["a"]), t0) * dzs + dzm).reshape(z_t.shape[0], HORIZON, N_SPATIAL, PATCH_DIM)

    def diff_pred(z_t, gseed):
        ztn = (patchify(z_t) - gzm) / gzs
        a_c = g_fou(act_h["a"]); B = z_t.shape[0]
        n_steps = 50; stride = max(DIFFUSION_STEPS // n_steps, 1)
        ts = list(reversed(list(range(0, DIFFUSION_STEPS, stride))[:n_steps]))
        gen = torch.Generator(device=device).manual_seed(gseed)
        x = torch.randn(B, HORIZON * N_SPATIAL, PATCH_DIM, device=device, generator=gen)
        for i, tv in enumerate(ts):
            t = torch.full((B,), tv, device=device, dtype=torch.long)
            px0 = g_dit(x, ztn, a_c, t)
            if cfg_w != 1.0:
                pu = g_dit(x, ztn, torch.zeros_like(a_c), t); px0 = pu + cfg_w * (px0 - pu)
            at = alphas[tv]; ap = alphas[ts[i + 1]] if i < len(ts) - 1 else torch.tensor(1.0, device=device)
            nd = (x - torch.sqrt(at) * px0) / torch.sqrt(1 - at + 1e-8)
            x = torch.sqrt(ap) * px0 + torch.sqrt(1 - ap) * nd
        return (x * gzs + gzm).reshape(B, HORIZON, N_SPATIAL, PATCH_DIM)

    # ---- channel diagnostic + calibration stats (diffusion vs real GT latents) over the manifold ----
    z_t_all = torch.tensor(np.stack([latents[i] for i in picks]), device=device)
    act_h = {"a": torch.tensor(np.stack([
        np.stack([[steers[i + kk], accels[i + kk]] for kk in range(HORIZON)]) for i in picks
    ]), dtype=torch.float32, device=device)}
    results = {"n_windows": len(picks), "horizons": hs, "cfg_w": cfg_w,
               "diffusion_ckpt": diffusion_ckpt, "window_manifest": picks, "by_horizon": {}}

    methods = ["direct", "diffusion_raw", "diffusion_calib", "vae_gt"]
    for h in hs:
        fids = {m: FrechetInceptionDistance(normalize=True).to(device) for m in methods}
        kids = {m: KernelInceptionDistance(normalize=True, subset_size=min(50, len(picks))).to(device) for m in methods}
        # channel-calibration stats for this horizon: per-channel (4) mean/std of diffusion-pred vs real-GT latents
        gt_lat = torch.tensor(np.stack([latents[i + 1 + h] for i in picks]), device=device)  # (W,4,32,32)
        # compute diffusion preds in batches; collect latents for calibration + images for FID
        ch_real_mean = gt_lat.mean(dim=(0, 2, 3)); ch_real_std = gt_lat.std(dim=(0, 2, 3))
        diff_lat_list = []
        bs = 8
        with torch.no_grad():
            for b in range(0, len(picks), bs):
                zt = z_t_all[b:b + bs]
                act_h["a"] = torch.tensor(np.stack([
                    np.stack([[steers[i + kk], accels[i + kk]] for kk in range(HORIZON)]) for i in picks[b:b + bs]
                ]), dtype=torch.float32, device=device)
                gp = unpatchify(diff_pred(zt, 1000 + b)[:, h])  # (b,4,32,32)
                diff_lat_list.append(gp)
            diff_lat = torch.cat(diff_lat_list, 0)
        ch_diff_mean = diff_lat.mean(dim=(0, 2, 3)); ch_diff_std = diff_lat.std(dim=(0, 2, 3))
        results["by_horizon"][str(h)] = {
            "ch_real_mean": [round(x, 4) for x in ch_real_mean.tolist()],
            "ch_diff_mean": [round(x, 4) for x in ch_diff_mean.tolist()],
            "ch_real_std": [round(x, 4) for x in ch_real_std.tolist()],
            "ch_diff_std": [round(x, 4) for x in ch_diff_std.tolist()],
        }
        def calibrate(lat):  # per-channel match diffusion -> real GT channel stats (visualization calibration)
            m = ch_diff_mean.view(1, 4, 1, 1); s = ch_diff_std.view(1, 4, 1, 1)
            rm = ch_real_mean.view(1, 4, 1, 1); rs = ch_real_std.view(1, 4, 1, 1)
            return (lat - m) / (s + 1e-6) * rs + rm
        # accumulate FID/KID
        with torch.no_grad():
            for b in range(0, len(picks), bs):
                idxs = picks[b:b + bs]
                zt = z_t_all[b:b + bs]
                act_h["a"] = torch.tensor(np.stack([
                    np.stack([[steers[i + kk], accels[i + kk]] for kk in range(HORIZON)]) for i in idxs
                ]), dtype=torch.float32, device=device)
                real = torch.cat([load_rgb(i, h) for i in idxs], 0)
                gt_img = decode(torch.tensor(np.stack([latents[i + 1 + h] for i in idxs]), device=device))
                d_img = decode(unpatchify(direct_pred(zt)[:, h]))
                g_lat = unpatchify(diff_pred(zt, 1000 + b)[:, h])
                g_img = decode(g_lat); gc_img = decode(calibrate(g_lat))
                for m, fake in [("direct", d_img), ("diffusion_raw", g_img),
                                ("diffusion_calib", gc_img), ("vae_gt", gt_img)]:
                    fids[m].update(real, real=True); fids[m].update(fake.clamp(0, 1), real=False)
                    kids[m].update(real, real=True); kids[m].update(fake.clamp(0, 1), real=False)
        rec = results["by_horizon"][str(h)]
        for m in methods:
            kmean, kstd = kids[m].compute()
            rec[m] = {"fid": round(float(fids[m].compute()), 3),
                      "kid_mean": round(float(kmean), 5), "kid_std": round(float(kstd), 5)}
        print(f"[fid] h={h}: " + " | ".join(f"{m} FID={rec[m]['fid']} KID={rec[m]['kid_mean']}" for m in methods))

    with open(f"{OUT_DIR}/fid_eval_{diffusion_ckpt.replace('/', '_')}.json", "w") as f:
        json.dump(results, f, indent=2)
    vol.commit()
    print(json.dumps(results, indent=2))
    return results


@_decorator
def motion_eval(n_windows: int = 200, seed: int = 0):
    """Quantify temporal dynamics captured: mean decoded-frame change t+1->t+H for
    GT vs direct vs diffusion. Honest scoping of the 'static across horizon' caveat."""
    import numpy as np
    import torch
    import torch.nn.functional as F
    from diffusers import AutoencoderKL

    spec = importlib.util.spec_from_file_location("tv", "/root/train_dit_vae_modal.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    patchify, unpatchify = mod.patchify, mod.unpatchify
    AnchoredVAEDiT, FourierActionEmbedding = mod.AnchoredVAEDiT, mod.FourierActionEmbedding
    DIT_CONFIG, FOURIER_CONFIG = mod.DIT_CONFIG, mod.FOURIER_CONFIG
    PATCH_DIM, N_SPATIAL = mod.PATCH_DIM, mod.N_SPATIAL
    CosineNoiseSchedule = mod._define_noise_schedule()
    device = torch.device("cuda")
    schedule = CosineNoiseSchedule(n_steps=DIFFUSION_STEPS).to(device)
    alphas = schedule.alphas_cumprod
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device).eval()
    data = np.load(VAE_NPZ, allow_pickle=True)
    latents = data["vae_latents"].astype(np.float32)
    scenes, splits = data["scene_names"], data["splits"]
    steers, accels = data["steer_norms"].astype(np.float32), data["accel_norms"].astype(np.float32)
    test_idx = np.where(splits == "test")[0]
    picks = []
    for sc in np.unique(scenes[test_idx]):
        idx = test_idx[scenes[test_idx] == sc]
        if len(idx) > HORIZON:
            picks.append(int(idx[0]))
        if len(picks) >= n_windows:
            break

    ck = torch.load(CKPT_PATHS["diffusion"], map_location=device, weights_only=False)
    g_dit = AnchoredVAEDiT(horizon=HORIZON, n_spatial=N_SPATIAL, **DIT_CONFIG).to(device)
    g_fou = FourierActionEmbedding(**FOURIER_CONFIG).to(device)
    g_dit.load_state_dict(ck["dit"]); g_fou.load_state_dict(ck["fourier"])
    if "ema" in ck and ck["ema"]:
        for nm, p in g_dit.named_parameters():
            if nm in ck["ema"]: p.data.copy_(ck["ema"][nm].to(device))
    g_dit.eval(); g_fou.eval(); gzm, gzs = ck["z_mean"].to(device), ck["z_std"].to(device)
    ckd = torch.load(CKPT_PATHS["direct"], map_location=device, weights_only=False)
    d_dit = AnchoredVAEDiT(horizon=HORIZON, n_spatial=N_SPATIAL, **DIT_CONFIG).to(device)
    d_fou = FourierActionEmbedding(**FOURIER_CONFIG).to(device)
    d_dit.load_state_dict(ckd["dit"]); d_fou.load_state_dict(ckd["fourier"]); d_dit.eval(); d_fou.eval()
    dzm, dzs = ckd["z_mean"].to(device), ckd["z_std"].to(device)

    def decode(grid):
        return ((vae.decode(grid.clamp(-6, 6) / SCALING).sample.clamp(-1, 1)) + 1) / 2

    def tchange(frames):  # frames: list of (1,3,256,256); mean consecutive L2
        ds = [(frames[i + 1] - frames[i]).pow(2).mean().sqrt().item() for i in range(len(frames) - 1)]
        return float(np.mean(ds))

    show = [0, 4, 8, 12, 15]
    gt_c, d_c, g_c = [], [], []
    with torch.no_grad():
        for fi in picks:
            z_t = torch.tensor(latents[fi:fi + 1], device=device)
            act = torch.stack([torch.tensor([steers[fi + kk], accels[fi + kk]], device=device)
                               for kk in range(HORIZON)]).unsqueeze(0)
            zf = torch.tensor(latents[fi + 1:fi + 1 + HORIZON], device=device).unsqueeze(0)
            # direct
            ztn = (patchify(z_t) - dzm) / dzs
            zr = ztn.unsqueeze(1).expand(-1, HORIZON, -1, -1).reshape(1, HORIZON * N_SPATIAL, PATCH_DIM)
            dp = (d_dit(zr, ztn, d_fou(act), torch.zeros(1, dtype=torch.long, device=device)) * dzs + dzm).reshape(1, HORIZON, N_SPATIAL, PATCH_DIM)
            # diffusion
            ztng = (patchify(z_t) - gzm) / gzs; a_c = g_fou(act)
            ns = 50; stride = max(DIFFUSION_STEPS // ns, 1); ts = list(reversed(list(range(0, DIFFUSION_STEPS, stride))[:ns]))
            gen = torch.Generator(device=device).manual_seed(seed)
            x = torch.randn(1, HORIZON * N_SPATIAL, PATCH_DIM, device=device, generator=gen)
            for i, tv in enumerate(ts):
                t = torch.full((1,), tv, device=device, dtype=torch.long); px0 = g_dit(x, ztng, a_c, t)
                at = alphas[tv]; ap = alphas[ts[i + 1]] if i < len(ts) - 1 else torch.tensor(1.0, device=device)
                nd = (x - torch.sqrt(at) * px0) / torch.sqrt(1 - at + 1e-8); x = torch.sqrt(ap) * px0 + torch.sqrt(1 - ap) * nd
            gp = (x * gzs + gzm).reshape(1, HORIZON, N_SPATIAL, PATCH_DIM)
            gt_c.append(tchange([decode(zf[:, k]) for k in show]))
            d_c.append(tchange([decode(unpatchify(dp[:, k])) for k in show]))
            g_c.append(tchange([decode(unpatchify(gp[:, k])) for k in show]))
    res = {"n_windows": len(picks), "steps": show,
           "gt_temporal_change": round(float(np.mean(gt_c)), 4),
           "direct_temporal_change": round(float(np.mean(d_c)), 4),
           "diffusion_temporal_change": round(float(np.mean(g_c)), 4)}
    res["diffusion_frac_of_gt"] = round(res["diffusion_temporal_change"] / (res["gt_temporal_change"] + 1e-9), 3)
    res["direct_frac_of_gt"] = round(res["direct_temporal_change"] / (res["gt_temporal_change"] + 1e-9), 3)
    print(f"[motion] GT={res['gt_temporal_change']} direct={res['direct_temporal_change']} "
          f"diffusion={res['diffusion_temporal_change']} | diff_frac_gt={res['diffusion_frac_of_gt']}")
    with open(f"{OUT_DIR}/motion_eval.json", "w") as f:
        json.dump(res, f, indent=2)
    vol.commit()
    return res


@_decorator
def frontier_eval(n_windows: int = 400, horizon_step: int = 15, n_calib: int = 200, seed: int = 0,
                  diffusion_ckpt: str = "diffusion"):
    """A1+B1: empirical distortion–perception curve over real operating points + a DEPLOYABLE
    per-channel calibration (estimated on TRAIN, not future-GT) + labeled post-hoc interpolation.
    Each point reports distortion (per-token CosSim↑ to GT) and perception (FID/KID↓ vs real RGB).
    diffusion_ckpt can be a CKPT_PATHS key or a full volume path (P1 capacity variants)."""
    import numpy as np
    import torch
    import torch.nn.functional as F
    from diffusers import AutoencoderKL
    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchmetrics.image.kid import KernelInceptionDistance
    from PIL import Image
    import torchvision.transforms.functional as TF

    spec = importlib.util.spec_from_file_location("tv", "/root/train_dit_vae_modal.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    patchify, unpatchify = mod.patchify, mod.unpatchify
    AnchoredVAEDiT, FourierActionEmbedding = mod.AnchoredVAEDiT, mod.FourierActionEmbedding
    DIT_CONFIG, FOURIER_CONFIG = mod.DIT_CONFIG, mod.FOURIER_CONFIG
    PATCH_DIM, N_SPATIAL = mod.PATCH_DIM, mod.N_SPATIAL
    CosineNoiseSchedule = mod._define_noise_schedule()
    dev = torch.device("cuda")
    h = horizon_step
    sch = CosineNoiseSchedule(n_steps=DIFFUSION_STEPS).to(dev); alphas = sch.alphas_cumprod
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(dev).eval()
    os.makedirs(OUT_DIR, exist_ok=True)
    data = np.load(VAE_NPZ, allow_pickle=True)
    lat = data["vae_latents"].astype(np.float32); scenes = data["scene_names"]; splits = data["splits"]
    steers = data["steer_norms"].astype(np.float32); accels = data["accel_norms"].astype(np.float32)
    ipaths = data["image_paths"] if "image_paths" in data else None

    def windows(split, cap, need_rgb=True):
        idx_all = np.where(splits == split)[0]; out = []
        for sc in np.unique(scenes[idx_all]):
            idx = idx_all[scenes[idx_all] == sc]
            for j in range(len(idx) - HORIZON):
                fi = int(idx[j])
                if (not need_rgb) or (ipaths is not None and fi + 1 + h < len(ipaths)
                                      and os.path.exists(f"{VOL_PATH}/nuscenes/{ipaths[fi+1+h]}")):
                    out.append(fi); break  # 1 per scene
            if len(out) >= cap: break
        return out[:cap]

    def load(tag):
        ck = torch.load(CKPT_PATHS.get(tag, tag), map_location=dev, weights_only=False)
        nb = int(ck.get("n_blocks", DIT_CONFIG["n_blocks"]))  # P1: rebuild at the ckpt's depth
        dit = AnchoredVAEDiT(horizon=HORIZON, n_spatial=N_SPATIAL, **{**DIT_CONFIG, "n_blocks": nb}).to(dev)
        fo = FourierActionEmbedding(**FOURIER_CONFIG).to(dev)
        dit.load_state_dict(ck["dit"]); fo.load_state_dict(ck["fourier"])
        if "ema" in ck and ck["ema"]:
            for nm, p in dit.named_parameters():
                if nm in ck["ema"]: p.data.copy_(ck["ema"][nm].to(dev))
            for nm, p in fo.named_parameters():
                if nm in ck["ema"]: p.data.copy_(ck["ema"][nm].to(dev))
        dit.eval(); fo.eval(); return dit, fo, ck["z_mean"].to(dev), ck["z_std"].to(dev)

    d_dit, d_fo, dzm, dzs = load("direct"); g_dit, g_fo, gzm, gzs = load(diffusion_ckpt)

    def act_of(fis):
        return torch.tensor(np.stack([np.stack([[steers[i+k], accels[i+k]] for k in range(HORIZON)]) for i in fis]),
                            dtype=torch.float32, device=dev)
    def direct_lat(fis):
        zt = torch.tensor(np.stack([lat[i] for i in fis]), device=dev); ztn = (patchify(zt)-dzm)/dzs
        zr = ztn.unsqueeze(1).expand(-1, HORIZON, -1, -1).reshape(len(fis), HORIZON*N_SPATIAL, PATCH_DIM)
        t0 = torch.zeros(len(fis), dtype=torch.long, device=dev)
        return unpatchify((d_dit(zr, ztn, d_fo(act_of(fis)), t0)*dzs+dzm).reshape(len(fis), HORIZON, N_SPATIAL, PATCH_DIM)[:, h])
    def diff_lat(fis, w=1.0, gs=0):
        zt = torch.tensor(np.stack([lat[i] for i in fis]), device=dev); ztn = (patchify(zt)-gzm)/gzs
        a = g_fo(act_of(fis)); B = len(fis); ns = 50; st = max(DIFFUSION_STEPS//ns, 1)
        ts = list(reversed(list(range(0, DIFFUSION_STEPS, st))[:ns]))
        gen = torch.Generator(device=dev).manual_seed(1000+gs)
        x = torch.randn(B, HORIZON*N_SPATIAL, PATCH_DIM, device=dev, generator=gen)
        for i, tv in enumerate(ts):
            t = torch.full((B,), tv, device=dev, dtype=torch.long); px0 = g_dit(x, ztn, a, t)
            if w != 1.0:
                pu = g_dit(x, ztn, torch.zeros_like(a), t); px0 = pu + w*(px0-pu)
            at = alphas[tv]; ap = alphas[ts[i+1]] if i < len(ts)-1 else torch.tensor(1.0, device=dev)
            nd = (x - torch.sqrt(at)*px0)/torch.sqrt(1-at+1e-8); x = torch.sqrt(ap)*px0 + torch.sqrt(1-ap)*nd
        return unpatchify((x*gzs+gzm).reshape(B, HORIZON, N_SPATIAL, PATCH_DIM)[:, h])
    def decode(g): return ((vae.decode(g.clamp(-6,6)/SCALING).sample.clamp(-1,1))+1)/2
    def rgb(fis):
        out = []
        for i in fis:
            im = Image.open(f"{VOL_PATH}/nuscenes/{ipaths[i+1+h]}").convert("RGB")
            w0, h0 = im.size; c = min(w0, h0)
            im = im.crop(((w0-c)//2, (h0-c)//2, (w0-c)//2+c, (h0-c)//2+c)).resize((256, 256))
            out.append(TF.to_tensor(im).unsqueeze(0).to(dev))
        return torch.cat(out, 0)
    def cossim(pred_lat_grid, fis):
        gt = torch.tensor(np.stack([lat[i+1+h] for i in fis]), device=dev)
        return F.cosine_similarity(patchify(pred_lat_grid), patchify(gt), dim=-1).mean().item()

    # --- B1: deployable per-channel calib estimated on TRAIN (mean+std), NOT future-GT ---
    tr = windows("train", n_calib, need_rgb=False)
    pred_acc, tgt_acc = [], []
    bs = 8
    with torch.no_grad():
        for b in range(0, len(tr), bs):
            fis = tr[b:b+bs]; pred_acc.append(diff_lat(fis, 1.0, gs=b))
            tgt_acc.append(torch.tensor(np.stack([lat[i+1+h] for i in fis]), device=dev))
    p_all = torch.cat(pred_acc, 0); t_all = torch.cat(tgt_acc, 0)
    p_m = p_all.mean(dim=(0,2,3)); p_s = p_all.std(dim=(0,2,3)); t_m = t_all.mean(dim=(0,2,3)); t_s = t_all.std(dim=(0,2,3))
    def caltrain(g):  # deployable: standardize by TRAIN pred stats, rescale to TRAIN target stats
        return (g - p_m.view(1,4,1,1))/(p_s.view(1,4,1,1)+1e-6)*t_s.view(1,4,1,1) + t_m.view(1,4,1,1)
    print(f"[frontier] caltrain ch shift (t_m-p_m)={[round(x,3) for x in (t_m-p_m).tolist()]} "
          f"std ratio (t_s/p_s)={[round(x,3) for x in (t_s/p_s).tolist()]}")

    te = windows("test", n_windows, need_rgb=True)
    ops = ["direct", "diff_w1", "diff_w2", "diff_caltrain", "interp_0.5"]
    fids = {o: FrechetInceptionDistance(normalize=True).to(dev) for o in ops}
    kids = {o: KernelInceptionDistance(normalize=True, subset_size=min(50, len(te))).to(dev) for o in ops}
    cs = {o: [] for o in ops}
    with torch.no_grad():
        for b in range(0, len(te), bs):
            fis = te[b:b+bs]; real = rgb(fis)
            dl = direct_lat(fis); g1 = diff_lat(fis, 1.0, gs=b); g2 = diff_lat(fis, 2.0, gs=b)
            pts = {"direct": dl, "diff_w1": g1, "diff_w2": g2,
                   "diff_caltrain": caltrain(g1), "interp_0.5": 0.5*dl + 0.5*g1}
            for o, gl in pts.items():
                cs[o].append(cossim(gl, fis))
                img = decode(gl).clamp(0,1)
                fids[o].update(real, real=True); fids[o].update(img, real=False)
                kids[o].update(real, real=True); kids[o].update(img, real=False)
    res = {"n_windows": len(te), "horizon_step": h, "n_calib": len(tr),
           "caltrain_shift": [round(x,4) for x in (t_m-p_m).tolist()],
           "points": {}}
    for o in ops:
        km, ks = kids[o].compute()
        res["points"][o] = {"cossim": round(float(np.mean(cs[o])),4), "fid": round(float(fids[o].compute()),2),
                            "kid_mean": round(float(km),5), "kid_std": round(float(ks),5)}
        print(f"[frontier] {o:14s} CosSim={res['points'][o]['cossim']:.4f}  FID={res['points'][o]['fid']:.1f}  KID={res['points'][o]['kid_mean']:.4f}")
    res["diffusion_ckpt"] = diffusion_ckpt
    ftag = "" if diffusion_ckpt == "diffusion" else "_" + diffusion_ckpt.strip("/").replace("/", "_")
    with open(f"{OUT_DIR}/frontier_eval{ftag}.json", "w") as f: json.dump(res, f, indent=2)
    vol.commit(); print(json.dumps(res, indent=2)); return res


@_decorator
def controllability(n_scenes: int = 40, horizon_step: int = 15, seed: int = 0, cfg_w: float = 1.0):
    """A2: does the generated future move WITH the conditioned steering? In-distribution steer
    sweep (train 5th/95th pct), fixed noise + scene; measure horizontal scene shift (1-D profile
    cross-correlation) vs the median-steer prediction; Spearman(steer, shift), diffusion vs direct.
    P4: cfg_w>1 applies classifier-free guidance in the diffusion sampler (uncond = zeroed action
    embedding, matching training force_zero) so we can trace rho(steer->shift) vs guidance scale."""
    import numpy as np
    import torch
    from scipy.stats import spearmanr
    from diffusers import AutoencoderKL

    spec = importlib.util.spec_from_file_location("tv", "/root/train_dit_vae_modal.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    patchify, unpatchify = mod.patchify, mod.unpatchify
    AnchoredVAEDiT, FourierActionEmbedding = mod.AnchoredVAEDiT, mod.FourierActionEmbedding
    DIT_CONFIG, FOURIER_CONFIG = mod.DIT_CONFIG, mod.FOURIER_CONFIG
    PATCH_DIM, N_SPATIAL = mod.PATCH_DIM, mod.N_SPATIAL
    CosineNoiseSchedule = mod._define_noise_schedule()
    dev = torch.device("cuda"); h = horizon_step
    sch = CosineNoiseSchedule(n_steps=DIFFUSION_STEPS).to(dev); alphas = sch.alphas_cumprod
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(dev).eval()
    os.makedirs(OUT_DIR, exist_ok=True)
    data = np.load(VAE_NPZ, allow_pickle=True)
    lat = data["vae_latents"].astype(np.float32); scenes = data["scene_names"]; splits = data["splits"]
    steers = data["steer_norms"].astype(np.float32); accels = data["accel_norms"].astype(np.float32)

    tr = np.where(splits == "train")[0]
    p5, p95 = float(np.percentile(steers[tr], 5)), float(np.percentile(steers[tr], 95))
    sweep = np.linspace(p5, p95, 5)
    print(f"[ctrl] in-distribution steer sweep (train p5..p95): {[round(x,3) for x in sweep]}")

    te = np.where(splits == "test")[0]; picks = []
    for sc in np.unique(scenes[te]):
        idx = te[scenes[te] == sc]
        if len(idx) > HORIZON: picks.append(int(idx[0]))
        if len(picks) >= n_scenes: break

    def load(tag):
        ck = torch.load(CKPT_PATHS[tag], map_location=dev, weights_only=False)
        dit = AnchoredVAEDiT(horizon=HORIZON, n_spatial=N_SPATIAL, **DIT_CONFIG).to(dev)
        fo = FourierActionEmbedding(**FOURIER_CONFIG).to(dev)
        dit.load_state_dict(ck["dit"]); fo.load_state_dict(ck["fourier"])
        if "ema" in ck and ck["ema"]:
            for nm, p in dit.named_parameters():
                if nm in ck["ema"]: p.data.copy_(ck["ema"][nm].to(dev))
        dit.eval(); fo.eval(); return dit, fo, ck["z_mean"].to(dev), ck["z_std"].to(dev)
    d_dit, d_fo, dzm, dzs = load("direct"); g_dit, g_fo, gzm, gzs = load("diffusion")

    def act_seq(fi, steer_val):
        a = np.stack([[steer_val, accels[fi+k]] for k in range(HORIZON)])
        return torch.tensor(a, dtype=torch.float32, device=dev).unsqueeze(0)
    def decode(g): return ((vae.decode(g.clamp(-6,6)/SCALING).sample.clamp(-1,1))+1)/2
    def hprofile(img):  # (1,3,256,256) -> (256,) horizontal intensity profile
        return img[0].mean(dim=(0,1)).detach().cpu().numpy()
    def hshift(prof, ref):  # peak of 1-D cross-correlation = horizontal shift (px)
        a = prof - prof.mean(); b = ref - ref.mean()
        cc = np.correlate(a, b, mode="full"); return int(cc.argmax() - (len(prof)-1))

    def predict_decode(kind, fi, steer_val):
        zt = torch.tensor(lat[fi:fi+1], device=dev)
        if kind == "direct":
            ztn = (patchify(zt)-dzm)/dzs
            zr = ztn.unsqueeze(1).expand(-1, HORIZON, -1, -1).reshape(1, HORIZON*N_SPATIAL, PATCH_DIM)
            t0 = torch.zeros(1, dtype=torch.long, device=dev)
            gl = unpatchify((d_dit(zr, ztn, d_fo(act_seq(fi, steer_val)), t0)*dzs+dzm).reshape(1, HORIZON, N_SPATIAL, PATCH_DIM)[:, h])
        else:
            ztn = (patchify(zt)-gzm)/gzs; a = g_fo(act_seq(fi, steer_val))
            ns = 50; st = max(DIFFUSION_STEPS//ns, 1); ts = list(reversed(list(range(0, DIFFUSION_STEPS, st))[:ns]))
            gen = torch.Generator(device=dev).manual_seed(seed)  # FIXED noise across steer values
            x = torch.randn(1, HORIZON*N_SPATIAL, PATCH_DIM, device=dev, generator=gen)
            for i, tv in enumerate(ts):
                t = torch.full((1,), tv, device=dev, dtype=torch.long); px0 = g_dit(x, ztn, a, t)
                if cfg_w != 1.0:
                    pu = g_dit(x, ztn, torch.zeros_like(a), t); px0 = pu + cfg_w * (px0 - pu)
                at = alphas[tv]; ap = alphas[ts[i+1]] if i < len(ts)-1 else torch.tensor(1.0, device=dev)
                nd = (x - torch.sqrt(at)*px0)/torch.sqrt(1-at+1e-8); x = torch.sqrt(ap)*px0 + torch.sqrt(1-ap)*nd
            gl = unpatchify((x*gzs+gzm).reshape(1, HORIZON, N_SPATIAL, PATCH_DIM)[:, h])
        return decode(gl)

    out = {"n_scenes": len(picks), "horizon_step": h, "cfg_w": cfg_w,
           "steer_sweep": [round(x,4) for x in sweep.tolist()],
           "train_p5": round(p5,4), "train_p95": round(p95,4)}
    with torch.no_grad():
        for kind in ["direct", "diffusion"]:
            rhos = []
            for fi in picks:
                ref = hprofile(predict_decode(kind, fi, float(sweep[len(sweep)//2])))  # median-steer reference
                shifts = [hshift(hprofile(predict_decode(kind, fi, float(s))), ref) for s in sweep]
                if np.std(shifts) > 0:
                    rho, _ = spearmanr(sweep, shifts); rhos.append(rho)
            rhos = [r for r in rhos if not np.isnan(r)]
            out[kind] = {"mean_spearman_steer_to_shift": round(float(np.mean(rhos)),4) if rhos else None,
                         "n_valid": len(rhos),
                         "frac_monotone_correct": round(float(np.mean([r > 0 for r in rhos])),3) if rhos else None}
            print(f"[ctrl] {kind}: mean Spearman(steer,shift)={out[kind]['mean_spearman_steer_to_shift']} "
                  f"(n={out[kind]['n_valid']}, frac_positive={out[kind]['frac_monotone_correct']})")
    ctag = "" if cfg_w == 1.0 else f"_cfg{cfg_w}"
    with open(f"{OUT_DIR}/controllability{ctag}.json", "w") as f: json.dump(out, f, indent=2)
    vol.commit(); print(json.dumps(out, indent=2)); return out


@_decorator
def planning_probe(n_scenes: int = 30, horizon_step: int = 15, seed: int = 0):
    """P5 (Future Work; NON-CIRCULAR): inverse-control / planning-primitive diagnostic.
    (a) Inverse-control on HELD-OUT targets: build a steer->shift grid (noise seed A); targets are the
        shifts at interpolated steers s* NOT in the grid, generated with a DIFFERENT noise seed B; invert
        the grid (linear fit) to pick s_hat; selection MAE=|s_hat-s*|. Run for diffusion AND direct on the
        SAME scenes -- the CONTRAST (invertible for the monotone diffusion model, ~chance for direct) is the
        non-circular signal; held-out s* + cross-seed target prevent the 'recover-what-you-put-in' tautology.
    NOTE: we measure inverse-control on the model's OWN counterfactual shifts, NOT pixel-fidelity to a single
    logged future. A 'reproduce-the-exact-GT-future' metric is ill-posed for a diffusion world model whose
    value is DISTRIBUTIONAL realism (CosSim~0.26 by design), not per-pixel MSE to one future -- it would favor
    the blurry conditional mean (the same perception-distortion trap). Preliminary (N=n_scenes); a planning
    PRIMITIVE diagnostic, not closed-loop planning accuracy."""
    import numpy as np
    import torch
    from scipy.stats import spearmanr
    from diffusers import AutoencoderKL

    spec = importlib.util.spec_from_file_location("tv", "/root/train_dit_vae_modal.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    patchify, unpatchify = mod.patchify, mod.unpatchify
    AnchoredVAEDiT, FourierActionEmbedding = mod.AnchoredVAEDiT, mod.FourierActionEmbedding
    DIT_CONFIG, FOURIER_CONFIG = mod.DIT_CONFIG, mod.FOURIER_CONFIG
    PATCH_DIM, N_SPATIAL = mod.PATCH_DIM, mod.N_SPATIAL
    CosineNoiseSchedule = mod._define_noise_schedule()
    dev = torch.device("cuda"); h = horizon_step
    sch = CosineNoiseSchedule(n_steps=DIFFUSION_STEPS).to(dev); alphas = sch.alphas_cumprod
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(dev).eval()
    os.makedirs(OUT_DIR, exist_ok=True)
    data = np.load(VAE_NPZ, allow_pickle=True)
    lat = data["vae_latents"].astype(np.float32); scenes = data["scene_names"]; splits = data["splits"]
    steers = data["steer_norms"].astype(np.float32); accels = data["accel_norms"].astype(np.float32)

    tr = np.where(splits == "train")[0]
    p5, p95 = float(np.percentile(steers[tr], 5)), float(np.percentile(steers[tr], 95))
    grid = np.linspace(p5, p95, 5)                  # the steer grid used to build the inversion map
    targets = (grid[:-1] + grid[1:]) / 2.0          # HELD-OUT interpolated steers (midpoints), 4 values
    rng = np.random.default_rng(seed)
    chance_mae = float(np.mean([np.mean(np.abs(rng.uniform(p5, p95, 4000) - s)) for s in targets]))  # random-pick baseline
    print(f"[plan] grid={[round(x,3) for x in grid]} held-out targets={[round(x,3) for x in targets]} chance_MAE={chance_mae:.3f}")

    te = np.where(splits == "test")[0]; picks = []
    for sc in np.unique(scenes[te]):
        idx = te[scenes[te] == sc]
        if len(idx) > HORIZON: picks.append(int(idx[0]))
        if len(picks) >= n_scenes: break

    def load(tag):
        ck = torch.load(CKPT_PATHS[tag], map_location=dev, weights_only=False)
        dit = AnchoredVAEDiT(horizon=HORIZON, n_spatial=N_SPATIAL, **DIT_CONFIG).to(dev)
        fo = FourierActionEmbedding(**FOURIER_CONFIG).to(dev)
        dit.load_state_dict(ck["dit"]); fo.load_state_dict(ck["fourier"])
        if "ema" in ck and ck["ema"]:
            for nm, p in dit.named_parameters():
                if nm in ck["ema"]: p.data.copy_(ck["ema"][nm].to(dev))
        dit.eval(); fo.eval(); return dit, fo, ck["z_mean"].to(dev), ck["z_std"].to(dev)
    d_dit, d_fo, dzm, dzs = load("direct"); g_dit, g_fo, gzm, gzs = load("diffusion")
    MODELS = {"direct": (d_dit, d_fo, dzm, dzs), "diffusion": (g_dit, g_fo, gzm, gzs)}

    def decode(g): return ((vae.decode(g.clamp(-6,6)/SCALING).sample.clamp(-1,1))+1)/2
    def hprofile(img): return img[0].mean(dim=(0,1)).detach().cpu().numpy()
    def hshift(prof, ref):
        a = prof - prof.mean(); b = ref - ref.mean()
        cc = np.correlate(a, b, mode="full"); return int(cc.argmax() - (len(prof)-1))

    def seq_const(fi, steer_val):   # constant-steer sequence (real per-step accel)
        a = np.stack([[steer_val, accels[fi+k]] for k in range(HORIZON)])
        return torch.tensor(a, dtype=torch.float32, device=dev).unsqueeze(0)

    def predict(kind, fi, act_t, noise_seed):
        dit, fo, zm, zs = MODELS[kind]
        zt = torch.tensor(lat[fi:fi+1], device=dev); ztn = (patchify(zt)-zm)/zs; a = fo(act_t)
        if kind == "direct":
            zr = ztn.unsqueeze(1).expand(-1, HORIZON, -1, -1).reshape(1, HORIZON*N_SPATIAL, PATCH_DIM)
            t0 = torch.zeros(1, dtype=torch.long, device=dev)
            gl = unpatchify((dit(zr, ztn, a, t0)*zs+zm).reshape(1, HORIZON, N_SPATIAL, PATCH_DIM)[:, h])
        else:
            ns = 50; st = max(DIFFUSION_STEPS//ns, 1); ts = list(reversed(list(range(0, DIFFUSION_STEPS, st))[:ns]))
            gen = torch.Generator(device=dev).manual_seed(noise_seed)
            x = torch.randn(1, HORIZON*N_SPATIAL, PATCH_DIM, device=dev, generator=gen)
            for i, tv in enumerate(ts):
                t = torch.full((1,), tv, device=dev, dtype=torch.long); px0 = dit(x, ztn, a, t)
                at = alphas[tv]; ap = alphas[ts[i+1]] if i < len(ts)-1 else torch.tensor(1.0, device=dev)
                nd = (x - torch.sqrt(at)*px0)/torch.sqrt(1-at+1e-8); x = torch.sqrt(ap)*px0 + torch.sqrt(1-ap)*nd
            gl = unpatchify((x*gzs+gzm).reshape(1, HORIZON, N_SPATIAL, PATCH_DIM)[:, h])
        return decode(gl)

    out = {"n_scenes": len(picks), "horizon_step": h,
           "steer_grid": [round(x,4) for x in grid.tolist()],
           "heldout_targets": [round(x,4) for x in targets.tolist()],
           "inverse_control_chance_mae": round(chance_mae,4),
           "note": "PRELIMINARY Future-Work probe; planning-primitive diagnostic, not closed-loop planning."}
    with torch.no_grad():
        for kind in ["direct", "diffusion"]:
            inv_maes, inv_r2s, inv_rhos = [], [], []
            for fi in picks:
                # --- (a) inverse-control on held-out targets ---
                ref = hprofile(predict(kind, fi, seq_const(fi, float(grid[len(grid)//2])), seed))  # grid ref, seed A
                gshift = np.array([hshift(hprofile(predict(kind, fi, seq_const(fi, float(s)), seed)), ref) for s in grid], float)
                if np.std(gshift) > 0:
                    rho, _ = spearmanr(grid, gshift); inv_rhos.append(rho)
                    A = np.vstack([grid, np.ones_like(grid)]).T
                    slope, intercept = np.linalg.lstsq(A, gshift, rcond=None)[0]
                    pred = slope*grid + intercept; ss_res = np.sum((gshift-pred)**2); ss_tot = np.sum((gshift-gshift.mean())**2)
                    inv_r2s.append(float(1 - ss_res/ss_tot) if ss_tot > 0 else 0.0)
                    if abs(slope) > 1e-6:
                        errs = []
                        for s_star in targets:
                            dstar = hshift(hprofile(predict(kind, fi, seq_const(fi, float(s_star)), seed+1)), ref)  # seed B
                            s_hat = float(np.clip((dstar - intercept)/slope, p5, p95))
                            errs.append(abs(s_hat - s_star))
                        inv_maes.append(float(np.mean(errs)))
            out[kind] = {
                "inverse_control": {
                    "mae_steer": round(float(np.mean(inv_maes)),4) if inv_maes else None,
                    "mae_vs_chance_ratio": round(float(np.mean(inv_maes))/chance_mae,3) if inv_maes else None,
                    "mean_grid_R2": round(float(np.mean(inv_r2s)),3) if inv_r2s else None,
                    "mean_grid_spearman": round(float(np.mean(inv_rhos)),3) if inv_rhos else None,
                    "n_valid": len(inv_maes),
                    "metric": "held-out interpolated target steers + cross-seed; s_hat via linear inversion of the steer->shift grid; MAE=|s_hat-s*| vs random-pick chance"}}
            ic = out[kind]["inverse_control"]
            print(f"[plan] {kind}: inv-ctrl MAE={ic['mae_steer']} (chance={chance_mae:.3f}, ratio={ic['mae_vs_chance_ratio']}, "
                  f"R2={ic['mean_grid_R2']}, rho={ic['mean_grid_spearman']}, n_valid={ic['n_valid']})")
    with open(f"{OUT_DIR}/planning_probe.json", "w") as f: json.dump(out, f, indent=2)
    vol.commit(); print(json.dumps(out, indent=2)); return out


@_decorator
def rollout_eval(n_windows: int = 30, seed: int = 0):
    """P2: 2-chunk autoregressive rollout. Chunk1: z_t -> z_{t+1..t+16}. Chunk2: feed predicted z_{t+16}
    back as z_t -> z_{t+17..t+32}, using the REAL GT actions a_{t+16:t+32} (critical). Per chunk endpoint
    (t+16, t+32) report FID vs real RGB + CosSim-to-GT + sharpness, for diffusion vs direct. Tests whether
    diffusion stays on-manifold over rollout while regression compounds blur. Models trained single-pass all-H."""
    import numpy as np
    import torch
    import torch.nn.functional as F
    from diffusers import AutoencoderKL
    from torchmetrics.image.fid import FrechetInceptionDistance
    from PIL import Image
    import torchvision.transforms.functional as TF

    spec = importlib.util.spec_from_file_location("tv", "/root/train_dit_vae_modal.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    patchify, unpatchify = mod.patchify, mod.unpatchify
    AnchoredVAEDiT, FourierActionEmbedding = mod.AnchoredVAEDiT, mod.FourierActionEmbedding
    DIT_CONFIG, FOURIER_CONFIG = mod.DIT_CONFIG, mod.FOURIER_CONFIG
    PATCH_DIM, N_SPATIAL = mod.PATCH_DIM, mod.N_SPATIAL
    CosineNoiseSchedule = mod._define_noise_schedule()
    dev = torch.device("cuda")
    sch = CosineNoiseSchedule(n_steps=DIFFUSION_STEPS).to(dev); alphas = sch.alphas_cumprod
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(dev).eval()
    os.makedirs(OUT_DIR, exist_ok=True)
    data = np.load(VAE_NPZ, allow_pickle=True)
    lat = data["vae_latents"].astype(np.float32); scenes = data["scene_names"]; splits = data["splits"]
    steers = data["steer_norms"].astype(np.float32); accels = data["accel_norms"].astype(np.float32)
    ipaths = data["image_paths"] if "image_paths" in data else None
    te = np.where(splits == "test")[0]

    picks = []
    for sc in np.unique(scenes[te]):
        idx = te[scenes[te] == sc]
        for j in range(len(idx)):
            fi = int(idx[j])
            need = [fi + 16, fi + 32]
            if fi + 32 < int(idx[-1]) + 1 and ipaths is not None and all(
                    n < len(ipaths) and os.path.exists(f"{VOL_PATH}/nuscenes/{ipaths[n]}") for n in need):
                picks.append(fi); break
        if len(picks) >= n_windows: break
    print(f"[rollout] {len(picks)} windows (need t+32 in-scene + RGB)")

    def load(tag):
        ck = torch.load(CKPT_PATHS[tag], map_location=dev, weights_only=False)
        dit = AnchoredVAEDiT(horizon=HORIZON, n_spatial=N_SPATIAL, **DIT_CONFIG).to(dev)
        fo = FourierActionEmbedding(**FOURIER_CONFIG).to(dev)
        dit.load_state_dict(ck["dit"]); fo.load_state_dict(ck["fourier"])
        if "ema" in ck and ck["ema"]:
            for nm, p in dit.named_parameters():
                if nm in ck["ema"]: p.data.copy_(ck["ema"][nm].to(dev))
        dit.eval(); fo.eval(); return dit, fo, ck["z_mean"].to(dev), ck["z_std"].to(dev)
    d_dit, d_fo, dzm, dzs = load("direct"); g_dit, g_fo, gzm, gzs = load("diffusion")

    def aseq(fi0):  # real actions a_{fi0 .. fi0+15}
        return torch.tensor(np.stack([[steers[fi0 + k], accels[fi0 + k]] for k in range(HORIZON)]),
                            dtype=torch.float32, device=dev).unsqueeze(0)
    def decode(g): return ((vae.decode(g.clamp(-6, 6) / SCALING).sample.clamp(-1, 1)) + 1) / 2
    def sharp(img01):
        gray = img01.mean(1, keepdim=True)
        ker = torch.tensor([[0,1,0],[1,-4,1],[0,1,0]], dtype=torch.float32, device=dev).view(1,1,3,3)
        return F.conv2d(gray, ker, padding=1).view(img01.shape[0], -1).var(dim=1)
    def rgb(n):
        im = Image.open(f"{VOL_PATH}/nuscenes/{ipaths[n]}").convert("RGB"); w0,h0 = im.size; c = min(w0,h0)
        im = im.crop(((w0-c)//2,(h0-c)//2,(w0-c)//2+c,(h0-c)//2+c)).resize((256,256))
        return TF.to_tensor(im).unsqueeze(0).to(dev)

    def predict_full(kind, z_grid, fi0):  # returns (H,4,32,32) predicted latents from z_grid + real actions a_{fi0..}
        zt = z_grid.unsqueeze(0)
        if kind == "direct":
            ztn = (patchify(zt)-dzm)/dzs
            zr = ztn.unsqueeze(1).expand(-1,HORIZON,-1,-1).reshape(1,HORIZON*N_SPATIAL,PATCH_DIM)
            t0 = torch.zeros(1,dtype=torch.long,device=dev)
            tok = (d_dit(zr, ztn, d_fo(aseq(fi0)), t0)*dzs+dzm).reshape(1,HORIZON,N_SPATIAL,PATCH_DIM)[0]
        else:
            ztn = (patchify(zt)-gzm)/gzs; a = g_fo(aseq(fi0)); ns=50; st=max(DIFFUSION_STEPS//ns,1)
            ts = list(reversed(list(range(0,DIFFUSION_STEPS,st))[:ns]))
            gen = torch.Generator(device=dev).manual_seed(seed); x = torch.randn(1,HORIZON*N_SPATIAL,PATCH_DIM,device=dev,generator=gen)
            for i,tv in enumerate(ts):
                t = torch.full((1,),tv,device=dev,dtype=torch.long); px0 = g_dit(x,ztn,a,t)
                at=alphas[tv]; ap=alphas[ts[i+1]] if i<len(ts)-1 else torch.tensor(1.0,device=dev)
                nd=(x-torch.sqrt(at)*px0)/torch.sqrt(1-at+1e-8); x=torch.sqrt(ap)*px0+torch.sqrt(1-ap)*nd
            tok = (x*gzs+gzm).reshape(1,HORIZON,N_SPATIAL,PATCH_DIM)[0]
        return torch.stack([unpatchify(tok[k:k+1])[0] for k in range(HORIZON)], 0)  # (H,4,32,32)

    res = {"n_windows": len(picks), "chunks": ["t+16", "t+32"], "models": {}}
    for kind in ["direct", "diffusion"]:
        fid16 = FrechetInceptionDistance(normalize=True).to(dev); fid32 = FrechetInceptionDistance(normalize=True).to(dev)
        cs = {"t+16": [], "t+32": []}; sh = {"t+16": [], "t+32": []}
        with torch.no_grad():
            for fi in picks:
                zt = torch.tensor(lat[fi], device=dev)
                ch1 = predict_full(kind, zt, fi)                 # z_{t+1..t+16}, actions a_{fi..fi+15}
                z16 = ch1[15]
                ch2 = predict_full(kind, z16, fi + 16)           # z_{t+17..t+32}, REAL actions a_{fi+16..fi+31}
                z32 = ch2[15]
                for tag, zp, gt_n in [("t+16", z16, fi+16), ("t+32", z32, fi+32)]:
                    img = decode(zp.unsqueeze(0)).clamp(0,1); real = rgb(gt_n)
                    (fid16 if tag=="t+16" else fid32).update(real, real=True)
                    (fid16 if tag=="t+16" else fid32).update(img, real=False)
                    gt = torch.tensor(lat[gt_n], device=dev)
                    cs[tag].append(F.cosine_similarity(patchify(zp.unsqueeze(0)), patchify(gt.unsqueeze(0)), dim=-1).mean().item())
                    sh[tag].append(sharp(img).item())
        res["models"][kind] = {
            "fid": {"t+16": round(float(fid16.compute()),2), "t+32": round(float(fid32.compute()),2)},
            "cossim": {t: round(float(np.mean(cs[t])),4) for t in cs},
            "sharpness": {t: round(float(np.mean(sh[t])),4) for t in sh}}
        m = res["models"][kind]
        print(f"[rollout] {kind}: FID t+16={m['fid']['t+16']} t+32={m['fid']['t+32']} | CosSim {m['cossim']} | sharp {m['sharpness']}")
    with open(f"{OUT_DIR}/rollout_eval.json", "w") as f: json.dump(res, f, indent=2)
    vol.commit(); print(json.dumps(res, indent=2)); return res


@_decorator
def framerate_preflight(n_windows: int = 600, k_nn: int = 12, seed: int = 0):
    """P3 (data-only): does conditional multimodality grow at lower effective frame rate?
    TRUE within-horizon temporal subsampling on VAE latents: future = z_{t+s}, z_{t+2s}, ..., z_{t+H*s}
    for stride s in {1,2,4} (H=16 -> spans 16/32/64 keyframes ≈ 8/16/32 s). Scene-controlled spread of the
    farthest future z_{t+H*s} among pooled-z_t nearest-neighbors. Gate: spread(s=4)/spread(s=1) > 1.2×."""
    import numpy as np
    import torch
    import torch.nn.functional as F

    dev = torch.device("cuda")
    data = np.load(VAE_NPZ, allow_pickle=True)
    lat = data["vae_latents"].astype(np.float32); scenes = data["scene_names"]; splits = data["splits"]
    steers = data["steer_norms"].astype(np.float32)
    te = np.where(splits == "test")[0]
    rng = np.random.default_rng(seed)

    def l2n(x): return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)
    out = {"strides": {}, "k_nn": k_nn}
    for s in [1, 2, 4]:
        z_t, z_last, steer_std = [], [], []
        for sc in np.unique(scenes[te]):
            idx = te[scenes[te] == sc]
            for j in range(len(idx)):
                fut = idx[j + s: j + s * (HORIZON + 1): s]   # within-horizon subsample (the fix)
                if len(fut) < HORIZON: break
                z_t.append(lat[idx[j]]); z_last.append(lat[fut[-1]])
                steer_std.append(float(steers[fut].std()))
        if len(z_t) < 100:
            print(f"[framerate] stride={s}: too few windows ({len(z_t)}), skip"); continue
        zt = np.stack(z_t).reshape(len(z_t), -1); zl = np.stack(z_last).reshape(len(z_last), -1)
        ss = np.array(steer_std)
        if n_windows and len(zt) > n_windows:
            sel = rng.choice(len(zt), n_windows, replace=False); zt, zl, ss = zt[sel], zl[sel], ss[sel]
        pool = torch.tensor(l2n(zt), device=dev); fut = torch.tensor(l2n(zl), device=dev)
        hv = np.where(ss >= np.quantile(ss, 0.75))[0]
        sims = pool[hv] @ pool.T
        spreads, copies = [], []
        for r, a in enumerate(hv):
            row = sims[r].clone(); row[a] = -2
            nn = torch.topk(row, k_nn).indices
            spreads.append((1 - (fut[nn] @ fut[a])).mean().item())          # future spread among scene-NN
            copies.append((1 - F.cosine_similarity(pool[a:a+1], fut[a:a+1])).item())  # 1-cos(z_t,z_last) proxy
        out["strides"][str(s)] = {"seconds_ahead": round(HORIZON * s * 0.5, 1), "n_windows": len(zt),
                                  "n_hv": len(hv), "sigma_scene_future": round(float(np.mean(spreads)), 4),
                                  "copy_1_minus_cos": round(float(np.mean(copies)), 4)}
        print(f"[framerate] s={s} ({out['strides'][str(s)]['seconds_ahead']}s): "
              f"sigma_scene_future={out['strides'][str(s)]['sigma_scene_future']} "
              f"copy(1-cos)={out['strides'][str(s)]['copy_1_minus_cos']}")
    if "1" in out["strides"] and "4" in out["strides"]:
        r = out["strides"]["4"]["sigma_scene_future"] / (out["strides"]["1"]["sigma_scene_future"] + 1e-9)
        out["spread_ratio_s4_over_s1"] = round(r, 3); out["gate_pass_emerging"] = bool(r > 1.2)
        print(f"[framerate] spread ratio s4/s1 = {r:.3f} -> emerging={out['gate_pass_emerging']}")
    with open(f"{OUT_DIR}/framerate_preflight.json", "w") as f: json.dump(out, f, indent=2)
    vol.commit(); print(json.dumps(out, indent=2)); return out


def _entry(fn):
    return app.local_entrypoint()(fn) if app is not None else fn


@_entry
def main(task: str = "eval", models: str = "diffusion", k: int = 8, cfg_weights: str = "1.0",
         n_windows: int = 48, steps_eval: str = "3,15", n_fig: int = 5, cfg_w: float = 1.0,
         wps: int = 1, diffusion_ckpt: str = "diffusion"):
    if task == "figure":
        res = make_figure.remote(n_fig, cfg_w)
        print(json.dumps(res, indent=2))
        return
    if task == "action_use":
        res = action_use.remote(n_windows)
        out = Path("artifacts/full/gen_eval_action_use.json")
        if not out.parent.exists():
            out = Path("code/latent-world-models-av/artifacts/full/gen_eval_action_use.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(res, indent=2))
        print(json.dumps(res, indent=2)); print(f"Saved {out}")
        return
    if task == "rollout":
        res = rollout_eval.remote(n_windows)
        out = Path("artifacts/full/rollout_eval.json")
        if not out.parent.exists():
            out = Path("code/latent-world-models-av/artifacts/full/rollout_eval.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(res, indent=2)); print(f"Saved {out}")
        return
    if task == "framerate":
        res = framerate_preflight.remote(n_windows)
        out = Path("artifacts/full/framerate_preflight.json")
        if not out.parent.exists():
            out = Path("code/latent-world-models-av/artifacts/full/framerate_preflight.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(res, indent=2)); print(f"Saved {out}")
        return
    if task == "controllability":
        res = controllability.remote(n_windows, 15, 0, cfg_w)
        ctag = "" if cfg_w == 1.0 else f"_cfg{cfg_w}"
        out = Path(f"artifacts/full/controllability{ctag}.json")
        if not out.parent.exists():
            out = Path(f"code/latent-world-models-av/artifacts/full/controllability{ctag}.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(res, indent=2)); print(f"Saved {out}")
        return
    if task == "planning":
        res = planning_probe.remote(n_windows)
        out = Path("artifacts/full/planning_probe.json")
        if not out.parent.exists():
            out = Path("code/latent-world-models-av/artifacts/full/planning_probe.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(res, indent=2)); print(json.dumps(res, indent=2)); print(f"Saved {out}")
        return
    if task == "frontier":
        res = frontier_eval.remote(n_windows, 15, 200, 0, diffusion_ckpt)
        ftag = "" if diffusion_ckpt == "diffusion" else "_" + diffusion_ckpt.strip("/").replace("/", "_")
        out = Path(f"artifacts/full/frontier_eval{ftag}.json")
        if not out.parent.exists():
            out = Path(f"code/latent-world-models-av/artifacts/full/frontier_eval{ftag}.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(res, indent=2)); print(f"Saved {out}")
        return
    if task == "motion":
        res = motion_eval.remote(n_windows)
        out = Path("artifacts/full/motion_eval.json")
        if not out.parent.exists():
            out = Path("code/latent-world-models-av/artifacts/full/motion_eval.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(res, indent=2)); print(json.dumps(res, indent=2)); print(f"Saved {out}")
        return
    if task == "fid":
        res = fid_eval.remote(n_windows, steps_eval, cfg_w, 0, diffusion_ckpt, wps)
        tag = diffusion_ckpt.strip("/").replace("/", "_") if "/" in diffusion_ckpt else diffusion_ckpt
        out = Path(f"artifacts/full/fid_eval_{tag}.json")
        if not out.parent.exists():
            out = Path(f"code/latent-world-models-av/artifacts/full/fid_eval_{tag}.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(res, indent=2))
        print(f"Saved {out}")
        return
    res = gen_eval.remote(models, k, cfg_weights, n_windows, steps_eval, n_fig)
    tag = models.replace(",", "_")
    out = Path(f"artifacts/full/gen_eval_{tag}.json")
    if not out.parent.exists():
        out = Path(f"code/latent-world-models-av/artifacts/full/gen_eval_{tag}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    print(f"Saved {out}")
