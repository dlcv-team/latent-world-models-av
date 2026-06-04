"""Motion-mini: a compact chain-anchor JUMP world-model that visibly MOVES (t+4..t+16).

Closes the loop "a compact model actually predicts motion". Diagnosis: the canonical DiT predicts every future
token as a delta from the SAME present z_t (-> re-render present + texture, not accumulated ego-motion). Fix here:
a JUMP transition objective (z_t -> z_{t+4}, horizon=1) + per-step re-anchoring at inference (open-loop chain of 4
jumps to t+16), on AVG-POOLED 16x16 latents (no re-encode), with light anchor-noise augmentation for covariate-shift
robustness. Stage A (direct) is the deliverable; sharpness refine is a separate later step.

SEPARATE names everywhere (*_motionmini_jump4_*); NEVER touches canonical seed_0 / cfg0.1 result JSON.

Tasks:
  modal run scripts/train_motion_mini_modal.py --task preflight --n-scenes 8
  modal run scripts/train_motion_mini_modal.py --task train --epochs 15 --tag smoke   # M0
  modal run scripts/train_motion_mini_modal.py --task train --epochs 40 --tag full    # M1
  modal run scripts/train_motion_mini_modal.py --task eval  --tag full --n-scenes 40
  modal run scripts/train_motion_mini_modal.py --task render --tag full --scene-ids 3217,438,3336
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
    app = modal.App("lwm-av-motion-mini")
    vol = modal.Volume.from_name("nuscenes-full")
    image = (modal.Image.debian_slim(python_version="3.12")
             .apt_install("ffmpeg")
             .pip_install("torch==2.5.1", "torchvision==0.20.1", "numpy==1.26.4", "Pillow>=10.0",
                          "diffusers==0.31.0", "transformers==4.46.3", "accelerate==1.1.1",
                          "matplotlib>=3.8", "scipy>=1.11", "imageio>=2.34", "imageio-ffmpeg>=0.4.9",
                          "scikit-image>=0.22", "opencv-python-headless>=4.8")
             .add_local_file(str(TRAIN_SCRIPT), remote_path="/root/train_dit_vae_modal.py"))
else:
    app = vol = image = None

VOL_PATH = "/vol"
SPATIAL_DIR = f"{VOL_PATH}/embeddings/spatial"
VAE_NPZ = f"{SPATIAL_DIR}/sd_vae_latents.npz"
CKPT_DIR = f"{VOL_PATH}/dits/vae_latent"
DIRECT_FULL = f"{CKPT_DIR}/h16/seed_0/dit.pt"          # canonical full-res direct (baseline + read-only)
MM_DIR = f"{CKPT_DIR}/motionmini_jump4"                 # our separate ckpt dir
OUT = f"{VOL_PATH}/viz"
SCALING = 0.18215
# motion-mini constants (own; do NOT import parent 32/64).
# NOTE: avg-pooling 32->16 FAILED preflight (SSIM 0.36 decode, lowfreq ratio 0.61); horizon=1 already gives the
# 64-token speed win, so we train at NATIVE 32x32 (clean decode, full motion, same cost). pool() becomes identity.
GRID, PATCH = 32, 4
PATCH_DIM = PATCH * PATCH * 4                            # 64
N_SPATIAL = (GRID // PATCH) ** 2                         # 64
MODEL_DIM, N_BLOCKS, N_HEADS = 192, 2, 4
DT, N_JUMP, HORIZON_SRC = 4, 4, 16                       # jump stride, #jumps to t+16, source horizon


def _dec(fn):
    return app.function(volumes={VOL_PATH: vol}, image=image, gpu="A10G", timeout=7200, memory=32768)(fn) if app else fn


def _load_mod():
    spec = importlib.util.spec_from_file_location("tv", "/root/train_dit_vae_modal.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def _common():
    """Shared setup: torch, VAE, data, pooled latents, helpers. Returns a dict."""
    import numpy as np, torch
    import torch.nn.functional as F
    import torchvision.transforms.functional as TF
    from diffusers import AutoencoderKL
    mod = _load_mod()
    dev = torch.device("cuda"); torch.set_grad_enabled(False)
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(dev).eval()
    Sched = mod._define_noise_schedule(); alphas = Sched(n_steps=1000).to(dev).alphas_cumprod  # for diffusion mode
    data = np.load(VAE_NPZ, allow_pickle=True)
    lat = data["vae_latents"].astype(np.float32)
    scenes, splits = data["scene_names"], data["splits"]
    steers, accels = data["steer_norms"].astype(np.float32), data["accel_norms"].astype(np.float32)
    acts = np.stack([steers, accels], axis=-1).astype(np.float32)  # (N,2)
    def pool(z):  # (B,4,32,32)->(B,4,16,16)
        return F.avg_pool2d(z, PATCH if GRID == 16 else 1) if z.shape[-1] == 32 else z
    def pool_np(zt):  # numpy (.,4,32,32)->(.,4,16,16) via torch
        return F.avg_pool2d(torch.tensor(zt), 2).numpy()
    def decode(zl):  # (B,4,G,G) scaled latent -> [0,1] RGB tensor
        return ((vae.decode(zl.clamp(-6, 6) / SCALING).sample.clamp(-1, 1)) + 1) / 2
    def to_u8(t):
        return (t[0].permute(1, 2, 0).clamp(0, 1).detach().cpu().numpy() * 255).astype("uint8")
    # ---- motion metrics (ported from motion_fidelity) ----
    BLUR_KS, BLUR_SIGMA = 31, 8.0
    def _low(img): return TF.gaussian_blur(img, BLUR_KS, [BLUR_SIGMA, BLUR_SIGMA])
    def _rms(a): return float(a.pow(2).mean().sqrt().item())
    def seq_lowhigh(fr):
        lo = [_low(f) for f in fr]
        L = float(np.mean([_rms(lo[i + 1] - lo[i]) for i in range(len(fr) - 1)]))
        H = float(np.mean([_rms((fr[i + 1] - lo[i + 1]) - (fr[i] - lo[i])) for i in range(len(fr) - 1)]))
        return L, H
    def _prof(img, ax): return img[0].mean(dim=(0, ax)).detach().cpu().numpy()
    def _sh1(p, r):
        a = p - p.mean(); b = r - r.mean(); cc = np.correlate(a, b, "full"); return int(cc.argmax() - (len(p) - 1))
    def disp_seq(fr): return [(_sh1(_prof(fr[i + 1], 1), _prof(fr[i], 1)), _sh1(_prof(fr[i + 1], 2), _prof(fr[i], 2)))
                              for i in range(len(fr) - 1)]
    def dir_corr(pr, gt):
        cs = [(px * gx + py * gy) / (((gx * gx + gy * gy) ** 0.5) * ((px * px + py * py) ** 0.5))
              for (px, py), (gx, gy) in zip(pr, gt) if (gx * gx + gy * gy) > 0 and (px * px + py * py) > 0]
        return float(np.mean(cs)) if cs else 0.0
    def disp_mag(s): return float(np.mean([(dx * dx + dy * dy) ** 0.5 for dx, dy in s]))
    def sharp(img):
        g = img.mean(1, keepdim=True)
        ker = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]], device=dev).view(1, 1, 3, 3)
        return float(F.conv2d(g, ker, padding=1).var().item())
    return dict(np=np, torch=torch, F=F, mod=mod, dev=dev, vae=vae, alphas=alphas, lat=lat, scenes=scenes, splits=splits,
                acts=acts, pool=pool, pool_np=pool_np, decode=decode, to_u8=to_u8,
                seq_lowhigh=seq_lowhigh, disp_seq=disp_seq, dir_corr=dir_corr, disp_mag=disp_mag, sharp=sharp)


def _build_windows(C, split):
    """Windows fi where fi..fi+16 same scene. Returns list of fi (ints)."""
    np = C["np"]; scenes, splits = C["scenes"], C["splits"]
    idx = np.where(splits == split)[0]
    out = []
    for sc in np.unique(scenes[idx]):
        s = np.sort(idx[scenes[idx] == sc]); sset = set(s.tolist())
        for a in s.tolist():
            if a + HORIZON_SRC < len(scenes) and scenes[a + HORIZON_SRC] == sc and (a + HORIZON_SRC) in sset:
                out.append(int(a))
    return out


def _make_model(C, ckpt=None):
    mod, dev = C["mod"], C["dev"]
    dit = mod.AnchoredVAEDiT(horizon=1, n_spatial=N_SPATIAL, patch_dim=PATCH_DIM, model_dim=MODEL_DIM,
                             n_blocks=N_BLOCKS, n_heads=N_HEADS, mlp_ratio=4.0, dropout=0.0).to(dev)
    fou = mod.FourierActionEmbedding(action_dim=2, n_frequencies=64, base=2.0, out_dim=MODEL_DIM).to(dev)
    if ckpt is not None:
        ck = C["torch"].load(ckpt, map_location=dev, weights_only=False)
        dit.load_state_dict(ck["dit"]); fou.load_state_dict(ck["fourier"])
        if ck.get("ema"):
            for nm, p in list(dit.named_parameters()) + list(fou.named_parameters()):
                if nm in ck["ema"]: p.data.copy_(ck["ema"][nm].to(dev))
        dit.eval(); fou.eval()
        return dit, fou, ck["z_mean"].to(dev), ck["z_std"].to(dev), ck.get("mode", "direct")
    return dit, fou, None, None, "direct"


def _act_emb(C, fou, fi):
    """Aggregate a_{fi:fi+DT} (DT CAN steps) -> (1,1,MODEL_DIM)."""
    torch = C["torch"]; dev = C["dev"]
    seg = torch.tensor(C["acts"][fi:fi + DT], device=dev).unsqueeze(0)  # (1,DT,2)
    return fou(seg).mean(dim=1, keepdim=True)                            # (1,DT,Md)->(1,1,Md)


@_dec
def preflight(n_scenes: int = 8):
    """Phase-0: pooled-16x16 vs full-res GT motion ratio + SSIM (decode readability)."""
    import json
    from skimage.metrics import structural_similarity as ssim
    C = _common(); np = C["np"]; torch = C["torch"]
    picks = _build_windows(C, "test")[:n_scenes]
    ratios, ssims = [], []
    for fi in picks:
        full = [C["decode"](torch.tensor(C["lat"][fi + DT * j:fi + DT * j + 1], device=C["dev"])) for j in range(N_JUMP + 1)]
        pooled = [C["decode"](C["pool"](torch.tensor(C["lat"][fi + DT * j:fi + DT * j + 1], device=C["dev"]))) for j in range(N_JUMP + 1)]
        Lf, _ = C["seq_lowhigh"](full); Lp, _ = C["seq_lowhigh"](pooled)
        ratios.append(Lp / (Lf + 1e-9))
        a = C["to_u8"](full[0]); b = C["to_u8"](torch.nn.functional.interpolate(pooled[0], size=a.shape[0], mode="bilinear"))
        ssims.append(float(ssim(a, b, channel_axis=2)))
    res = {"n": len(picks), "lowfreq_ratio_pooled_over_full": round(float(np.mean(ratios)), 3),
           "ssim_pooled_vs_full": round(float(np.mean(ssims)), 3),
           "gate_ratio>=0.70": bool(np.mean(ratios) >= 0.70), "gate_ssim>=0.85": bool(np.mean(ssims) >= 0.85)}
    print("[preflight]", json.dumps(res))
    return res


@_dec
def train(epochs: int = 40, n_windows: int = 8000, anchor_noise: float = 0.0, lam_res: float = 1.0,
          motion_pow: float = 0.5, lr: float = 2e-4, batch: int = 256, tag: str = "full", seed: int = 0,
          mode: str = "direct"):
    """JUMP training (teacher-forced anchors). mode=direct (regression, blurry-moving) or diffusion (sharp x0-pred)."""
    import json, time
    C = _common(); np = C["np"]; torch = C["torch"]; F = C["F"]; dev = C["dev"]; alphas = C["alphas"]
    torch.set_grad_enabled(True)
    mod = C["mod"]; patchify = mod.patchify
    wins = _build_windows(C, "train")
    rng = np.random.default_rng(seed); rng.shuffle(wins)
    wins = wins[:n_windows]
    # build jump tuples (anchor fi+4j, target fi+4(j+1)) + per-tuple latent-motion weight
    A, B_, segs, mscore = [], [], [], []
    lat = C["lat"]
    for fi in wins:
        for j in range(N_JUMP):
            a0, a1 = fi + DT * j, fi + DT * (j + 1)
            A.append(a0); B_.append(a1); segs.append(a0)
            mscore.append(float(np.sqrt(((lat[a1] - lat[a0]) ** 2).mean())))
    A, B_, segs, mscore = map(np.array, (A, B_, segs, mscore))
    w = (mscore / (mscore.mean() + 1e-9)) ** motion_pow; w = w / w.sum()
    # pooled+patchified normalized stats from a sample
    samp = C["pool"](torch.tensor(lat[A[:2000]], device=dev))
    flat = patchify(samp, PATCH).reshape(-1, PATCH_DIM)
    z_mean, z_std = flat.mean(0), flat.std(0).clamp(min=1e-6)
    dit, fou, _, _, _ = _make_model(C)
    n_par = sum(p.numel() for p in dit.parameters()) + sum(p.numel() for p in fou.parameters())
    print(f"[train tag={tag} mode={mode}] params={n_par/1e6:.2f}M tuples={len(A)} epochs={epochs} anchor_noise={anchor_noise}")
    opt = torch.optim.AdamW(list(dit.parameters()) + list(fou.parameters()), lr=lr, weight_decay=1e-4)
    ema = {nm: p.detach().clone() for nm, p in list(dit.named_parameters()) + list(fou.named_parameters())}
    def npatch(z): return (patchify(C["pool"](z), PATCH) - z_mean) / z_std
    steps_per = max(len(A) // batch, 1); t0 = time.time()
    for ep in range(epochs):
        tot = 0.0
        for _ in range(steps_per):
            bi = rng.choice(len(A), size=batch, p=w)
            za = torch.tensor(lat[A[bi]], device=dev); zt = torch.tensor(lat[B_[bi]], device=dev)
            zap = npatch(za); ztp = npatch(zt)                       # (B,64,16) normalized
            zan = zap + anchor_noise * torch.randn_like(zap)          # anchor-noise aug
            seg = torch.tensor(np.stack([C["acts"][s:s + DT] for s in segs[bi]]), device=dev)  # (B,DT,2)
            a_emb = fou(seg).mean(dim=1, keepdim=True)               # (B,1,Md)
            if mode == "diffusion":
                tt = torch.randint(0, 1000, (len(bi),), device=dev)
                ab = alphas[tt].view(-1, 1, 1)
                xn = torch.sqrt(ab) * ztp + torch.sqrt(1 - ab) * torch.randn_like(ztp)
                pred = dit(xn, zan, a_emb, tt)                        # x0-pred conditioned on (anchor, action)
                loss = F.mse_loss(pred, ztp)
            else:
                t0v = torch.zeros(len(bi), dtype=torch.long, device=dev)
                pred = dit(zan, zan, a_emb, t0v)                      # (B,64,Pd) = zan + delta
                loss = F.mse_loss(pred, ztp) + lam_res * F.mse_loss(pred - zap, ztp - zap)
            opt.zero_grad(); loss.backward(); opt.step(); tot += float(loss)
            with torch.no_grad():
                for nm, p in list(dit.named_parameters()) + list(fou.named_parameters()):
                    ema[nm].mul_(0.999).add_(p.detach(), alpha=0.001)
        print(f"  ep{ep+1}/{epochs} loss={tot/steps_per:.4f} ({time.time()-t0:.0f}s)")
    os.makedirs(f"{MM_DIR}/{tag}", exist_ok=True)
    torch.save({"dit": dit.state_dict(), "fourier": fou.state_dict(), "ema": ema,
                "z_mean": z_mean.cpu(), "z_std": z_std.cpu(), "grid": GRID, "patch": PATCH,
                "dt": DT, "n_jump": N_JUMP, "model_dim": MODEL_DIM, "n_blocks": N_BLOCKS, "n_params": n_par,
                "mode": mode},
               f"{MM_DIR}/{tag}/dit.pt")
    vol.commit(); print(f"[train] saved {MM_DIR}/{tag}/dit.pt  params={n_par/1e6:.2f}M")
    return {"tag": tag, "params_M": round(n_par / 1e6, 2), "tuples": len(A)}


def _jump_step(C, dit, fou, zmn, zsd, z_anchor, fi_step, mode="direct", nseed=0):
    torch = C["torch"]; mod = C["mod"]; dev = C["dev"]
    zp = (mod.patchify(z_anchor, PATCH) - zmn) / zsd
    a_emb = _act_emb(C, fou, fi_step)
    if mode == "diffusion":  # DDIM x0 sampling conditioned on the (moving) anchor + action
        alphas = C["alphas"]; ns = 25; st = max(1000 // ns, 1); ts = list(reversed(list(range(0, 1000, st))[:ns]))
        gen = torch.Generator(device=dev).manual_seed(nseed + int(fi_step))
        x = torch.randn(zp.shape, device=dev, generator=gen)
        for i, tv in enumerate(ts):
            t = torch.full((zp.shape[0],), tv, device=dev, dtype=torch.long)
            px0 = dit(x, zp, a_emb, t)
            at = alphas[tv]; ap = alphas[ts[i + 1]] if i < len(ts) - 1 else torch.tensor(1.0, device=dev)
            nd = (x - torch.sqrt(at) * px0) / torch.sqrt(1 - at + 1e-8); x = torch.sqrt(ap) * px0 + torch.sqrt(1 - ap) * nd
        pred = x
    else:
        pred = dit(zp, zp, a_emb, torch.zeros(zp.shape[0], dtype=torch.long, device=dev))
    return mod.unpatchify(pred * zsd + zmn, patch_size=PATCH, grid_h=GRID, grid_w=GRID)


def _jump_chain(C, dit, fou, zmn, zsd, fi, teacher_forced=False, mode="direct"):
    """OPEN-LOOP (own anchors) or TEACHER-FORCED (GT anchors) chain; returns 5 decoded frames t+0,4,8,12,16."""
    torch = C["torch"]
    z_curr = C["pool"](torch.tensor(C["lat"][fi:fi + 1], device=C["dev"]))
    frames = [C["decode"](z_curr)]
    for j in range(N_JUMP):
        anchor = C["pool"](torch.tensor(C["lat"][fi + DT * j:fi + DT * j + 1], device=C["dev"])) if teacher_forced else z_curr
        z_curr = _jump_step(C, dit, fou, zmn, zsd, anchor, fi + DT * j, mode=mode, nseed=j)
        frames.append(C["decode"](z_curr))
    return frames


@_dec
def eval(tag: str = "full", n_scenes: int = 40):
    """Open-loop jump-chain motion_fidelity (mined + ordinary held-out) + full-res-direct baseline."""
    import json
    C = _common(); np = C["np"]; torch = C["torch"]; mod = C["mod"]
    dit, fou, zmn, zsd, mmode = _make_model(C, f"{MM_DIR}/{tag}/dit.pt")
    test = _build_windows(C, "test")
    ordinary = test[:n_scenes]
    # mined heroes (high latent motion) for visual support
    msc = [float(np.sqrt(((C["lat"][fi + HORIZON_SRC] - C["lat"][fi]) ** 2).mean())) for fi in test]
    mined = [test[i] for i in np.argsort(msc)[::-1][:n_scenes]]

    def agg(picks, tf=False):
        dc, lf, dm = [], [], []
        for fi in picks:
            gt = [C["decode"](C["pool"](torch.tensor(C["lat"][fi + DT * j:fi + DT * j + 1], device=C["dev"]))) for j in range(N_JUMP + 1)]
            pr = _jump_chain(C, dit, fou, zmn, zsd, fi, teacher_forced=tf, mode=mmode)
            gtd = C["disp_seq"](gt); prd = C["disp_seq"](pr)
            gtlo, _ = C["seq_lowhigh"](gt); prlo, _ = C["seq_lowhigh"](pr)
            dc.append(C["dir_corr"](prd, gtd)); lf.append(prlo / (gtlo + 1e-9))
            dm.append(C["disp_mag"](prd) / (C["disp_mag"](gtd) + 1e-9))
        return {"n": len(picks), "disp_dir_corr": round(float(np.mean(dc)), 3),
                "lowfreq_frac": round(float(np.mean(lf)), 3), "disp_mag_frac": round(float(np.mean(dm)), 3)}

    # baseline: canonical full-res direct at t+4,8,12,16 (own GT, normalized metrics comparable)
    def baseline(picks):
        ck = torch.load(DIRECT_FULL, map_location=C["dev"], weights_only=False)
        nb = int(ck.get("n_blocks", 4))
        d = mod.AnchoredVAEDiT(horizon=16, n_spatial=64, patch_dim=64, model_dim=256, n_blocks=nb, n_heads=4).to(C["dev"])
        fo = mod.FourierActionEmbedding(action_dim=2, n_frequencies=64, base=2.0, out_dim=256).to(C["dev"])
        d.load_state_dict(ck["dit"]); fo.load_state_dict(ck["fourier"]); d.eval(); fo.eval()
        dzm, dzs = ck["z_mean"].to(C["dev"]), ck["z_std"].to(C["dev"])
        dc, lf, dm = [], [], []
        for fi in picks:
            zt = torch.tensor(C["lat"][fi:fi + 1], device=C["dev"]); ztn = (mod.patchify(zt) - dzm) / dzs
            zr = ztn.unsqueeze(1).expand(-1, 16, -1, -1).reshape(1, 16 * 64, 64)
            act = torch.tensor(np.stack([C["acts"][fi + k] for k in range(16)]), device=C["dev"]).unsqueeze(0)
            dp = (d(zr, ztn, fo(act), torch.zeros(1, dtype=torch.long, device=C["dev"])) * dzs + dzm).reshape(1, 16, 64, 64)
            pr = [C["decode"](torch.tensor(C["lat"][fi:fi + 1], device=C["dev"]))] + \
                 [C["decode"](mod.unpatchify(dp[:, k])) for k in (3, 7, 11, 15)]
            gt = [C["decode"](torch.tensor(C["lat"][fi + s:fi + s + 1], device=C["dev"])) for s in (0, 4, 8, 12, 16)]
            gtd = C["disp_seq"](gt); prd = C["disp_seq"](pr); gtlo, _ = C["seq_lowhigh"](gt); prlo, _ = C["seq_lowhigh"](pr)
            dc.append(C["dir_corr"](prd, gtd)); lf.append(prlo / (gtlo + 1e-9)); dm.append(C["disp_mag"](prd) / (C["disp_mag"](gtd) + 1e-9))
        return {"n": len(picks), "disp_dir_corr": round(float(np.mean(dc)), 3),
                "lowfreq_frac": round(float(np.mean(lf)), 3), "disp_mag_frac": round(float(np.mean(dm)), 3)}

    res = {"tag": tag, "mode": mmode, "protocol": "jump_chain", "dt": DT, "n_jumps": N_JUMP,
           "motionmini_ordinary_openloop": agg(ordinary, tf=False),
           "motionmini_ordinary_teacherforced": agg(ordinary, tf=True),
           "motionmini_mined_openloop": agg(mined, tf=False),
           "baseline_fullres_direct_ordinary": baseline(ordinary)}
    o = res["motionmini_ordinary_openloop"]
    res["gate_motion_PASS"] = bool(o["disp_dir_corr"] >= 0.22 and o["lowfreq_frac"] >= 0.5)
    res["gate_motion_STRONG"] = bool(o["disp_dir_corr"] >= 0.28)
    with open(f"{OUT}/motion_fidelity_motionmini_jump4_{tag}.json", "w") as f: json.dump(res, f, indent=2)
    vol.commit(); print("[eval]", json.dumps(res, indent=2)); return res


@_dec
def render(tag: str = "full", scene_ids: str = ""):
    """Rollout filmstrip (PNG) + mp4/gif of the open-loop jump chain on hero scenes."""
    import imageio.v2 as imageio
    from PIL import Image
    C = _common(); torch = C["torch"]
    dit, fou, zmn, zsd, mmode = _make_model(C, f"{MM_DIR}/{tag}/dit.pt")
    picks = [int(x) for x in scene_ids.split(",") if x.strip()] or _build_windows(C, "test")[:3]
    def up(img): return C["np"].asarray(Image.fromarray(img).resize((256, 256), Image.BILINEAR))
    made = []
    for fi in picks:
        gt = [C["to_u8"](C["decode"](torch.tensor(C["lat"][fi + DT * j:fi + DT * j + 1], device=C["dev"]))) for j in range(N_JUMP + 1)]
        pr = [C["to_u8"](f) for f in _jump_chain(C, dit, fou, zmn, zsd, fi)]
        sep = (C["np"].ones((pr[0].shape[0], 6, 3), "uint8") * 255)
        strip_gt = C["np"].concatenate(sum([[g, sep] for g in gt[:-1]], []) + [gt[-1]], axis=1)
        strip_pr = C["np"].concatenate(sum([[p, sep] for p in pr[:-1]], []) + [pr[-1]], axis=1)
        gap = C["np"].ones((6, strip_gt.shape[1], 3), "uint8") * 255
        imageio.imwrite(f"{OUT}/vmotion_mini_s{fi}_{tag}.png", C["np"].concatenate([strip_gt, gap, strip_pr], axis=0))
        imageio.mimwrite(f"{OUT}/vmotion_mini_s{fi}_{tag}.mp4", [up(p) for p in pr], fps=2, codec="libx264", quality=9, macro_block_size=1)
        made.append(fi)
    vol.commit(); print("[render] GT(top)/pred(bottom) strips + mp4 for", made); return {"made": made}


@_dec
def demo(tag: str = "smoke2", scene_ids: str = "3217,438,3336"):
    """V0 communication pack (no retrain): single-jump t+4 triptych + short t+0,4,8 strip + full5 + baseline + flow."""
    import json, imageio.v2 as imageio
    from PIL import Image
    from skimage.metrics import structural_similarity as ssim
    try:
        import cv2; HAS_CV2 = True
    except Exception:
        HAS_CV2 = False
    C = _common(); torch = C["torch"]; np = C["np"]; mod = C["mod"]; dev = C["dev"]
    dit, fou, zmn, zsd, mmode = _make_model(C, f"{MM_DIR}/{tag}/dit.pt")
    picks = [int(x) for x in scene_ids.split(",") if x.strip()]
    SZ = 256
    def u8(zl): return C["to_u8"](C["decode"](zl))
    def gtf(n): return torch.tensor(C["lat"][n:n + 1], device=dev)
    def up(img): return np.asarray(Image.fromarray(img).resize((SZ, SZ), Image.BILINEAR))
    def hstrip(frames):
        fr = [up(f) for f in frames]; s = np.ones((SZ, 6, 3), np.uint8) * 255
        return np.concatenate(sum([[f, s] for f in fr[:-1]], []) + [fr[-1]], axis=1)
    def stack(*rows):
        out = []; gap = np.ones((6, rows[0].shape[1], 3), np.uint8) * 255
        for i, r in enumerate(rows):
            out.append(r);
            if i < len(rows) - 1: out.append(gap)
        return np.concatenate(out, axis=0)
    def flow_rgb(a, b):
        ga = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY); gb = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY)
        fl = cv2.calcOpticalFlowFarneback(ga, gb, None, 0.5, 3, 21, 3, 5, 1.2, 0)
        mag, ang = cv2.cartToPolar(fl[..., 0], fl[..., 1]); hsv = np.zeros((a.shape[0], a.shape[1], 3), np.uint8)
        hsv[..., 0] = (ang * 90 / np.pi).astype(np.uint8); hsv[..., 1] = 255
        hsv[..., 2] = np.clip(mag * 16, 0, 255).astype(np.uint8); return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    # full-res direct baseline (5.4M, 16-joint) decoded at t+0,4,8,12,16
    ck = torch.load(DIRECT_FULL, map_location=dev, weights_only=False); nb = int(ck.get("n_blocks", 4))
    bd = mod.AnchoredVAEDiT(horizon=16, n_spatial=64, patch_dim=64, model_dim=256, n_blocks=nb, n_heads=4).to(dev)
    bfo = mod.FourierActionEmbedding(action_dim=2, n_frequencies=64, base=2.0, out_dim=256).to(dev)
    bd.load_state_dict(ck["dit"]); bfo.load_state_dict(ck["fourier"]); bd.eval(); bfo.eval()
    bzm, bzs = ck["z_mean"].to(dev), ck["z_std"].to(dev)
    def baseline_frames(fi):
        zt = gtf(fi); ztn = (mod.patchify(zt) - bzm) / bzs
        zr = ztn.unsqueeze(1).expand(-1, 16, -1, -1).reshape(1, 16 * 64, 64)
        act = torch.tensor(np.stack([C["acts"][fi + k] for k in range(16)]), device=dev).unsqueeze(0)
        dp = (bd(zr, ztn, bfo(act), torch.zeros(1, dtype=torch.long, device=dev)) * bzs + bzm).reshape(1, 16, 64, 64)
        return [u8(gtf(fi))] + [C["to_u8"](C["decode"](mod.unpatchify(dp[:, k]))) for k in (3, 7, 11, 15)]
    sc = {"tag": tag, "mode": mmode, "scenes": []}
    for fi in picks:
        present = u8(gtf(fi))
        z4 = _jump_step(C, dit, fou, zmn, zsd, gtf(fi), fi, mode=mmode)
        pred4, gt4 = u8(z4), u8(gtf(fi + 4))
        imageio.imwrite(f"{OUT}/demo_s{fi}_triptych_t4.png", hstrip([present, pred4, gt4]))  # present | pred t+4 | GT t+4
        chain = [C["to_u8"](f) for f in _jump_chain(C, dit, fou, zmn, zsd, fi, mode=mmode)]   # t+0,4,8,12,16
        gtseq = [u8(gtf(fi + 4 * j)) for j in range(5)]
        imageio.imwrite(f"{OUT}/demo_s{fi}_short_t048.png", stack(hstrip(gtseq[:3]), hstrip(chain[:3])))  # GT/pred t+0,4,8
        imageio.imwrite(f"{OUT}/demo_s{fi}_full5.png", stack(hstrip(gtseq), hstrip(chain)))               # supp: full + compounding
        imageio.imwrite(f"{OUT}/demo_s{fi}_baseline.png", stack(hstrip(gtseq), hstrip(baseline_frames(fi)), hstrip(chain)))  # GT/5.4M-direct/jump
        if HAS_CV2:
            fgt = [flow_rgb(gtseq[j], gtseq[j + 1]) for j in range(2)]; fpr = [flow_rgb(chain[j], chain[j + 1]) for j in range(2)]
            imageio.imwrite(f"{OUT}/demo_s{fi}_flow.png", stack(hstrip(fgt), hstrip(fpr)))  # flow GT(top)/pred(bot): t0->4, t4->8
        sc["scenes"].append({"fi": fi, "ssim_t4": round(float(ssim(pred4, gt4, channel_axis=2)), 3),
                             "sharp_pred_t4": round(C["sharp"](C["decode"](z4)), 5),
                             "sharp_gt_t4": round(C["sharp"](C["decode"](gtf(fi + 4))), 5),
                             "sharp_present": round(C["sharp"](C["decode"](gtf(fi))), 5)})
    with open(f"{OUT}/motion_demo_scorecard_{tag}.json", "w") as f: json.dump(sc, f, indent=2)
    vol.commit(); print("[demo]", json.dumps(sc)); return sc


@app.local_entrypoint() if app else (lambda f: f)
def main(task: str = "preflight", epochs: int = 40, n_windows: int = 8000, n_scenes: int = 40,
         anchor_noise: float = 0.0, tag: str = "full", scene_ids: str = "", mode: str = "direct"):
    if task == "preflight":
        print(preflight.remote(n_scenes))
    elif task == "train":
        print(train.remote(epochs, n_windows, anchor_noise, 1.0, 0.5, 2e-4, 256, tag, 0, mode))
    elif task == "eval":
        print(eval.remote(tag, n_scenes))
    elif task == "render":
        print(render.remote(tag, scene_ids))
    elif task == "demo":
        print(demo.remote(tag, scene_ids))
    else:
        raise SystemExit(f"unknown task {task}")
