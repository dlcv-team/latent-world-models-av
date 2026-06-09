"""Exp A: action counterfactual sensitivity (spatial tokens) + VAE decode figure.

Quantitative: per-token sensitivity = 1 - CosSim(pred(a_true), pred(a_cf)) on spatial
direct-anchored DiT vs MLP, with shuffle control and quality-under-true guard.

Visual: VAE-latent DiT decode for left vs right steer counterfactuals.

Usage::

    modal run scripts/eval_vae_counterfactual_modal.py
    modal run scripts/eval_vae_counterfactual_modal.py --encoder dino_vits14
"""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

from config import load_canonical

try:
    import modal
except ImportError:
    modal = None

if modal is not None:
    app = modal.App("lwm-av-counterfactual")
    vol = modal.Volume.from_name("nuscenes-full")
    _train_vae = Path(__file__).resolve().parent / "train_dit_vae_modal.py"
    image = (
        modal.Image.debian_slim(python_version="3.12")
        .pip_install(
            "torch==2.5.1", "numpy>=1.26", "Pillow>=10.0",
            "diffusers>=0.27", "matplotlib>=3.8", "accelerate", "transformers>=4.50",
            "torchvision>=0.20",
        )
        .add_local_file(str(_train_vae), remote_path="/root/train_dit_vae_modal.py")
    )
else:
    app = None
    vol = None
    image = None

VOL_PATH = "/vol"
SPATIAL_DIR = f"{VOL_PATH}/embeddings/spatial"
SPATIAL_CKPT = f"{VOL_PATH}/dits/spatial_anchored"
VAE_CKPT = f"{VOL_PATH}/dits/vae_latent"
VAE_NPZ = f"{SPATIAL_DIR}/sd_vae_latents.npz"
DATA_ROOT = f"{VOL_PATH}/nuscenes"

TARGET_DIM = 384
HORIZON = 16
STEER_PERTURB = 0.3
N_WINDOWS_DEFAULT = 200
MLP_HIDDEN = 512
MLP_EPOCHS = 50
TRAIN_BATCH = 64

SPATIAL_TOKENS = {"vit_s16": 49, "dino_vits14": 64}
DIT_CONFIG = {"z_dim": 384, "cond_dim": 384, "n_blocks": 4, "n_heads": 6, "mlp_ratio": 4.0, "dropout": 0.0}
MLP_CONFIG = {"z_dim": 384, "a_dim": 384, "hidden": MLP_HIDDEN, "dropout": 0.1}
FOURIER_CONFIG = {"n_frequencies": 64, "base": 2.0, "out_dim": 384}


def _decorator(fn):
    if app is not None:
        return app.function(
            volumes={VOL_PATH: vol}, image=image, gpu="A100", timeout=7200, memory=32768,
        )(fn)
    return fn


@_decorator
def run_counterfactual(
    encoder_name: str = "vit_s16",
    seed: int = 0,
    horizon: int = 16,
    n_windows: int = N_WINDOWS_DEFAULT,
    steer_perturb: float = STEER_PERTURB,
):
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from copy import deepcopy
    from matplotlib.backends.backend_pdf import PdfPages
    import matplotlib.pyplot as plt
    from PIL import Image
    import torchvision.transforms.functional as TF

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda")
    n_spatial = SPATIAL_TOKENS[encoder_name]

    # ---- inline models (match train_dit_spatial_anchored_modal.py) ----
    def _modulate(x, shift, scale):
        return x * (1.0 + scale) + shift

    class TimestepEmbedding(nn.Module):
        def __init__(self, cond_dim=384):
            super().__init__()
            self.cond_dim = cond_dim
            self.mlp = nn.Sequential(
                nn.Linear(cond_dim, cond_dim), nn.SiLU(), nn.Linear(cond_dim, cond_dim),
            )

        def forward(self, timestep):
            half_dim = self.cond_dim // 2
            freqs = torch.exp(
                -math.log(10000.0)
                * torch.arange(half_dim, device=timestep.device, dtype=torch.float32) / half_dim
            )
            args = timestep.float().unsqueeze(-1) * freqs.unsqueeze(0)
            return self.mlp(torch.cat([args.sin(), args.cos()], dim=-1))

    class DiTBlock(nn.Module):
        def __init__(self, dim=384, cond_dim=384, n_heads=6, mlp_ratio=4.0, dropout=0.0):
            super().__init__()
            self.norm_attn = nn.LayerNorm(dim, elementwise_affine=False)
            self.attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
            self.norm_mlp = nn.LayerNorm(dim, elementwise_affine=False)
            hidden = int(dim * mlp_ratio)
            self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))
            self.adaln_linear = nn.Linear(cond_dim, 6 * dim)

        def forward(self, x, cond):
            mod = self.adaln_linear(cond)
            shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = mod.chunk(6, dim=-1)
            h = _modulate(self.norm_attn(x), shift_a, scale_a)
            attn_out, _ = self.attn(h, h, h, need_weights=False)
            x = x + gate_a * attn_out
            h = _modulate(self.norm_mlp(x), shift_m, scale_m)
            x = x + gate_m * self.mlp(h)
            return x

    class AnchoredSpatialDiT(nn.Module):
        def __init__(self, z_dim=384, cond_dim=384, n_blocks=4, n_heads=6,
                     horizon=16, n_spatial=49, mlp_ratio=4.0, dropout=0.0):
            super().__init__()
            self.horizon = horizon
            self.n_spatial = n_spatial
            self.input_proj = nn.Linear(z_dim, z_dim)
            self.spatial_pos = nn.Parameter(torch.randn(1, n_spatial, z_dim) * 0.02)
            self.temporal_pos = nn.Parameter(torch.randn(1, horizon, z_dim) * 0.02)
            self.timestep_embed = TimestepEmbedding(cond_dim)
            self.z_t_proj = nn.Linear(z_dim, cond_dim)
            self.blocks = nn.ModuleList([
                DiTBlock(z_dim, cond_dim, n_heads, mlp_ratio, dropout) for _ in range(n_blocks)
            ])
            self.final_norm = nn.LayerNorm(z_dim, elementwise_affine=False)
            self.final_adaln = nn.Linear(cond_dim, 3 * z_dim)
            self.final_linear = nn.Linear(z_dim, z_dim)
            nn.init.zeros_(self.final_linear.weight)
            nn.init.zeros_(self.final_linear.bias)

        def forward(self, x_input, z_t_spatial, a_embed, timestep):
            B = x_input.shape[0]
            H, S, D = self.horizon, self.n_spatial, TARGET_DIM
            sp = self.spatial_pos.unsqueeze(1).expand(-1, H, -1, -1).reshape(1, H * S, D)
            tp = self.temporal_pos.unsqueeze(2).expand(-1, -1, S, -1).reshape(1, H * S, D)
            x = self.input_proj(x_input) + sp + tp
            z_t_pooled = z_t_spatial.mean(dim=1)
            cond_global = self.timestep_embed(timestep) + self.z_t_proj(z_t_pooled)
            a_broadcast = a_embed.unsqueeze(2).expand(-1, -1, S, -1).reshape(B, H * S, D)
            cond = cond_global.unsqueeze(1).expand(-1, H * S, -1) + a_broadcast
            for block in self.blocks:
                x = block(x, cond)
            mod = self.final_adaln(cond)
            shift, scale, gate = mod.chunk(3, dim=-1)
            delta = gate * self.final_linear(_modulate(self.final_norm(x), shift, scale))
            z_t_rep = z_t_spatial.unsqueeze(1).expand(-1, H, -1, -1).reshape(B, H * S, D)
            return z_t_rep + delta

    class FourierActionEmbedding(nn.Module):
        def __init__(self, action_dim=2, n_frequencies=64, base=2.0, out_dim=384):
            super().__init__()
            freqs = base ** torch.arange(n_frequencies, dtype=torch.float32) * torch.pi
            self.register_buffer("freqs", freqs)
            fourier_dim = action_dim * 2 * n_frequencies
            self.proj = nn.Sequential(
                nn.Linear(fourier_dim, out_dim), nn.GELU(), nn.Linear(out_dim, out_dim),
            )

        def forward(self, action):
            if action.dim() == 2:
                action = action.unsqueeze(1)
            x = action.unsqueeze(-1) * self.freqs
            x = torch.cat([x.sin(), x.cos()], dim=-1).flatten(-2)
            return self.proj(x)

    class SpatialMLPPredictor(nn.Module):
        def __init__(self, z_dim=384, a_dim=384, horizon=16, n_spatial=49, hidden=512, dropout=0.1):
            super().__init__()
            self.horizon = horizon
            self.n_spatial = n_spatial
            input_dim = z_dim + z_dim + a_dim
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(hidden, z_dim),
            )

        def forward(self, z_t_spatial, a_embed):
            B, S, D = z_t_spatial.shape
            H = self.horizon
            z_t_pool = z_t_spatial.mean(dim=1)
            outputs = []
            for h in range(H):
                a_h = a_embed[:, h, :].unsqueeze(1).expand(-1, S, -1)
                z_pool_exp = z_t_pool.unsqueeze(1).expand(-1, S, -1)
                x = torch.cat([z_t_spatial, z_pool_exp, a_h], dim=-1)
                outputs.append(z_t_spatial + self.net(x))
            return torch.stack(outputs, dim=1).reshape(B, H * S, D)

    # ---- load spatial data ----
    spatial_path = f"{SPATIAL_DIR}/{encoder_name}_spatial.npz"
    if not os.path.exists(spatial_path):
        print(f"ERROR: missing {spatial_path}")
        return None

    data = np.load(spatial_path, allow_pickle=True)
    spatial_emb = data["spatial_embeddings"]
    splits = data["splits"]
    steer_norms = data["steer_norms"]
    accel_norms = data["accel_norms"]
    scene_names = data["scene_names"]

    def build_windows(split_name, with_future=False):
        mask = splits == split_name
        emb = spatial_emb[mask]
        steers = steer_norms[mask]
        accels = accel_norms[mask]
        scenes = scene_names[mask]
        z_t_list, action_seq_list, zf_list = [], [], []
        for scene in np.unique(scenes):
            idx = np.where(scenes == scene)[0]
            for j in range(len(idx) - horizon):
                z_t_list.append(emb[idx[j]])
                action_seq_list.append(np.stack([
                    np.array([steers[idx[j + k]], accels[idx[j + k]]]) for k in range(horizon)
                ]))
                if with_future:
                    zf_list.append(emb[idx[j + 1: j + 1 + horizon]])
        out = (
            torch.tensor(np.array(z_t_list), dtype=torch.float32),
            torch.tensor(np.array(action_seq_list), dtype=torch.float32),
        )
        if with_future:
            out = out + (torch.tensor(np.array(zf_list), dtype=torch.float32),)
        return out

    z_t_train, act_train, zf_train = build_windows("train", with_future=True)
    z_t_test, act_test, zf_test = build_windows("test", with_future=True)

    train_steers = steer_norms[splits == "train"]
    p10, p90 = float(np.percentile(train_steers, 10)), float(np.percentile(train_steers, 90))
    print(f"[cf] train steer norm p10={p10:.3f} p90={p90:.3f} perturb={steer_perturb}")

    flat = z_t_train.reshape(-1, TARGET_DIM)
    z_mean = flat.mean(dim=0).to(device)
    z_std = flat.std(dim=0).clamp(min=1e-6).to(device)

    def normalize(x):
        return (x - z_mean) / z_std

    def denormalize(x):
        return x * z_std + z_mean

    # ---- load DiT checkpoint ----
    dit_ckpt_path = f"{SPATIAL_CKPT}/{encoder_name}/direct/h{horizon}/seed_{seed}/dit_checkpoint.pt"
    if not os.path.exists(dit_ckpt_path):
        print(f"ERROR: missing DiT ckpt {dit_ckpt_path}")
        return None

    ckpt = torch.load(dit_ckpt_path, map_location=device, weights_only=False)
    dit = AnchoredSpatialDiT(**DIT_CONFIG, horizon=horizon, n_spatial=n_spatial).to(device)
    fourier = FourierActionEmbedding(**FOURIER_CONFIG).to(device)
    dit.load_state_dict(ckpt["dit_state_dict"])
    ema = ckpt.get("ema_params", {})
    with torch.no_grad():
        for name, param in dit.named_parameters():
            if name in ema:
                param.copy_(ema[name])
        for name, param in fourier.named_parameters():
            if name in ema:
                param.copy_(ema[name])
    dit.eval()
    fourier.eval()

    def dit_predict(z_t_b, act_b):
        B = z_t_b.shape[0]
        z_t_rep = z_t_b.unsqueeze(1).expand(-1, horizon, -1, -1).reshape(B, horizon * n_spatial, TARGET_DIM)
        a_embed = fourier(act_b)
        t0 = torch.zeros((B,), device=device, dtype=torch.long)
        return dit(z_t_rep, z_t_b, a_embed, t0)

    # ---- train or load MLP (not saved by spatial train script) ----
    mlp_ckpt_path = f"{SPATIAL_CKPT}/{encoder_name}/direct/h{horizon}/seed_{seed}/mlp_checkpoint.pt"
    mlp = SpatialMLPPredictor(**MLP_CONFIG, horizon=horizon, n_spatial=n_spatial).to(device)
    fourier_mlp = FourierActionEmbedding(**FOURIER_CONFIG).to(device)

    if os.path.exists(mlp_ckpt_path):
        mlp_ckpt = torch.load(mlp_ckpt_path, map_location=device, weights_only=False)
        mlp.load_state_dict(mlp_ckpt["mlp_state_dict"])
        fourier_mlp.load_state_dict(mlp_ckpt["fourier_state_dict"])
        print(f"[cf] loaded MLP from {mlp_ckpt_path}")
    else:
        print(f"[cf] training MLP ({MLP_EPOCHS} epochs) for sensitivity baseline...")
        n_train = len(z_t_train)
        opt = torch.optim.Adam(list(mlp.parameters()) + list(fourier_mlp.parameters()), lr=1e-3)
        for epoch in range(MLP_EPOCHS):
            mlp.train()
            fourier_mlp.train()
            perm = torch.randperm(n_train)
            loss_sum, nb = 0.0, 0
            for start in range(0, n_train, TRAIN_BATCH * 2):
                end = min(start + TRAIN_BATCH * 2, n_train)
                idx = perm[start:end]
                B = len(idx)
                z_t_b = normalize(z_t_train[idx].to(device))
                act_b = act_train[idx].to(device)
                zf_b = normalize(zf_train[idx].to(device).reshape(B, horizon * n_spatial, TARGET_DIM))
                pred = mlp(z_t_b, fourier_mlp(act_b))
                loss = F.mse_loss(pred, zf_b)
                opt.zero_grad()
                loss.backward()
                opt.step()
                loss_sum += loss.item()
                nb += 1
            if epoch % 10 == 0 or epoch == MLP_EPOCHS - 1:
                print(f"  MLP epoch {epoch}: loss={loss_sum / max(nb, 1):.6f}")
        os.makedirs(os.path.dirname(mlp_ckpt_path), exist_ok=True)
        torch.save({
            "mlp_state_dict": mlp.state_dict(),
            "fourier_state_dict": fourier_mlp.state_dict(),
        }, mlp_ckpt_path)
        print(f"[cf] saved MLP to {mlp_ckpt_path}")

    mlp.eval()
    fourier_mlp.eval()

    def mlp_predict(z_t_b, act_b):
        return mlp(z_t_b, fourier_mlp(act_b))

    # ---- sensitivity on subset of test windows ----
    n_eval = min(n_windows, len(z_t_test))
    rng = np.random.default_rng(seed)
    eval_idx = rng.choice(len(z_t_test), size=n_eval, replace=False)

    def sensitivity_for(predict_fn, label):
        pert_sums = [0.0] * horizon
        shuffle_sums = [0.0] * horizon
        quality_sums = [0.0] * horizon
        total = 0
        with torch.no_grad():
            for i in eval_idx:
                z_t_b = normalize(z_t_test[i:i + 1].to(device))
                act_true = act_test[i:i + 1].to(device)
                act_pert = act_true.clone()
                act_pert[:, :, 0] = act_pert[:, :, 0] + steer_perturb
                perm_t = torch.randperm(horizon, device=device)
                act_shuf = act_true[:, perm_t, :]

                pred_true = denormalize(predict_fn(z_t_b, act_true)).reshape(1, horizon, n_spatial, TARGET_DIM)
                pred_pert = denormalize(predict_fn(z_t_b, act_pert)).reshape(1, horizon, n_spatial, TARGET_DIM)
                pred_shuf = denormalize(predict_fn(z_t_b, act_shuf)).reshape(1, horizon, n_spatial, TARGET_DIM)
                gt = zf_test[i:i + 1].to(device)

                for k in range(horizon):
                    cs_pert = F.cosine_similarity(pred_true[:, k], pred_pert[:, k], dim=-1).mean().item()
                    cs_shuf = F.cosine_similarity(pred_true[:, k], pred_shuf[:, k], dim=-1).mean().item()
                    cs_gt = F.cosine_similarity(pred_true[:, k], gt[:, k], dim=-1).mean().item()
                    quality_sums[k] += cs_gt
                    pert_sums[k] += 1.0 - cs_pert
                    shuffle_sums[k] += 1.0 - cs_shuf
                total += 1

        return {
            "model": label,
            "perturb_sensitivity_by_step": [round(s / total, 6) for s in pert_sums],
            "shuffle_sensitivity_by_step": [round(s / total, 6) for s in shuffle_sums],
            "quality_under_true_by_step": [round(s / total, 6) for s in quality_sums],
            "perturb_sensitivity_h16": round(pert_sums[-1] / total, 6),
            "shuffle_sensitivity_h16": round(shuffle_sums[-1] / total, 6),
            "quality_under_true_h16": round(quality_sums[-1] / total, 6),
            "n_windows": total,
        }

    dit_sens = sensitivity_for(dit_predict, "dit_spatial_direct")
    mlp_sens = sensitivity_for(mlp_predict, "mlp_spatial")
    ratio_h16 = (
        dit_sens["perturb_sensitivity_h16"] / mlp_sens["perturb_sensitivity_h16"]
        if mlp_sens["perturb_sensitivity_h16"] > 1e-8 else 0.0
    )

    gate_a = {
        "perturb_sens_h16_dit": dit_sens["perturb_sensitivity_h16"],
        "shuffle_sens_h16_dit": dit_sens["shuffle_sensitivity_h16"],
        "quality_h16_dit": dit_sens["quality_under_true_h16"],
        "pass_perturb_gt_005": dit_sens["perturb_sensitivity_h16"] > 0.05,
        "pass_perturb_gt_shuffle": dit_sens["perturb_sensitivity_h16"] > dit_sens["shuffle_sensitivity_h16"],
        "pass_quality_high": dit_sens["quality_under_true_h16"] > 0.75,
        "narrative_pass": (
            dit_sens["perturb_sensitivity_h16"] > 0.05
            and dit_sens["perturb_sensitivity_h16"] > dit_sens["shuffle_sensitivity_h16"]
            and dit_sens["quality_under_true_h16"] > 0.75
        ),
    }
    print(f"[cf] Gate A: {gate_a}")

    # ---- VAE visual counterfactual (import train module models) ----
    vae_figure_path = None
    if os.path.exists(VAE_NPZ) and os.path.exists(f"{VAE_CKPT}/h{horizon}/seed_{seed}/dit.pt"):
        import importlib.util
        spec = importlib.util.spec_from_file_location("vae_train", "/root/train_dit_vae_modal.py")
        if spec and spec.loader:
            vmod = importlib.util.module_from_spec(spec)
            try:
                spec.loader.exec_module(vmod)
            except Exception as e:
                print(f"[cf] VAE module load skipped: {e}")
                vmod = None
        else:
            vmod = None

        if vmod is not None:
            from diffusers import AutoencoderKL

            PATCH_DIM = vmod.PATCH_DIM
            N_SPATIAL_V = vmod.N_SPATIAL
            patchify = vmod.patchify
            unpatchify = vmod.unpatchify
            AnchoredVAEDiT = vmod.AnchoredVAEDiT
            FourierActionEmbedding = vmod.FourierActionEmbedding

            vdata = np.load(VAE_NPZ, allow_pickle=True)
            v_lat = vdata["vae_latents"]
            v_splits = vdata["splits"]
            v_steers = vdata["steer_norms"]
            v_accels = vdata["accel_norms"]
            v_scenes = vdata["scene_names"]

            v_ckpt = torch.load(f"{VAE_CKPT}/h{horizon}/seed_{seed}/dit.pt", map_location=device, weights_only=False)
            vz_mean = v_ckpt["z_mean"].to(device)
            vz_std = v_ckpt["z_std"].to(device)

            def vnorm(g):
                return (patchify(g) - vz_mean) / vz_std

            vdit = AnchoredVAEDiT(horizon=horizon, n_spatial=N_SPATIAL_V, **vmod.DIT_CONFIG).to(device)
            vfourier = FourierActionEmbedding(**vmod.FOURIER_CONFIG).to(device)
            vdit.load_state_dict(v_ckpt["dit"])
            vfourier.load_state_dict(v_ckpt["fourier"])
            ema_v = v_ckpt.get("ema", {})
            with torch.no_grad():
                for n, p in vdit.named_parameters():
                    if n in ema_v:
                        p.copy_(ema_v[n])
                for n, p in vfourier.named_parameters():
                    if n in ema_v:
                        p.copy_(ema_v[n])
            vdit.eval()
            vfourier.eval()
            vae_dec = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device).eval()
            scaling = load_canonical().vae_scaling_factor  # 0.18215 for runwayml/stable-diffusion-v1-5

            def vae_predict_tokens(z_grid, act_seq):
                z_n = vnorm(z_grid)
                B = z_n.shape[0]
                z_rep = z_n.unsqueeze(1).expand(-1, horizon, -1, -1).reshape(B, horizon * N_SPATIAL_V, PATCH_DIM)
                t0 = torch.zeros(B, dtype=torch.long, device=device)
                out = vdit(z_rep, z_n, vfourier(act_seq), t0)
                return (out * vz_std + vz_mean).reshape(B, horizon, N_SPATIAL_V, PATCH_DIM)

            test_vidx = np.where(v_splits == "test")[0]
            fig_dir = f"{SPATIAL_DIR}/counterfactual_figure"
            os.makedirs(fig_dir, exist_ok=True)
            steps_show = [4, 8, 12, 15]
            n_panels = 0
            with torch.no_grad():
                for sc in np.unique(v_scenes[test_vidx])[:3]:
                    idx = test_vidx[v_scenes[test_vidx] == sc]
                    if len(idx) <= horizon:
                        continue
                    frame_i = int(idx[len(idx) // 2])
                    z_t = torch.tensor(v_lat[frame_i:frame_i + 1], device=device)
                    act_base = torch.stack([
                        torch.tensor([v_steers[frame_i + k], v_accels[frame_i + k]], device=device)
                        for k in range(horizon)
                    ]).unsqueeze(0)
                    act_left = act_base.clone()
                    act_right = act_base.clone()
                    act_left[:, :, 0] = p10
                    act_right[:, :, 0] = p90
                    pred_l = vae_predict_tokens(z_t, act_left)
                    pred_r = vae_predict_tokens(z_t, act_right)
                    fig, axes = plt.subplots(2, len(steps_show), figsize=(2.2 * len(steps_show), 4.5))
                    for col, k in enumerate(steps_show):
                        for row, pred_tok in enumerate([pred_l, pred_r]):
                            pred_lat = unpatchify(pred_tok[:, k]).clamp(-3, 3)
                            img = vae_dec.decode(pred_lat / scaling).sample
                            im = ((img.clamp(-1, 1) + 1) / 2)[0].permute(1, 2, 0).cpu().numpy()
                            axes[row, col].imshow(im)
                            axes[row, col].axis("off")
                            axes[row, col].set_title(
                                f"t+{k} {'left' if row == 0 else 'right'}", fontsize=8
                            )
                    fig.suptitle(f"VAE counterfactual (steer p10 vs p90)", fontsize=9)
                    fig.savefig(f"{fig_dir}/panel_{n_panels}.png", dpi=110, bbox_inches="tight")
                    plt.close(fig)
                    n_panels += 1
                    if n_panels >= 2:
                        break

            pdf_path = f"{fig_dir}/action_counterfactual.pdf"
            with PdfPages(pdf_path) as pdf:
                for pi in range(n_panels):
                    img = np.array(Image.open(f"{fig_dir}/panel_{pi}.png"))
                    fig, ax = plt.subplots(figsize=(10, 4))
                    ax.imshow(img)
                    ax.axis("off")
                    pdf.savefig(fig, bbox_inches="tight")
                    plt.close(fig)
            vae_figure_path = pdf_path
            print(f"[cf] saved VAE figure {pdf_path}")

    result = {
        "encoder": encoder_name,
        "seed": seed,
        "horizon": horizon,
        "steer_perturb": steer_perturb,
        "steer_p10": p10,
        "steer_p90": p90,
        "dit": dit_sens,
        "mlp": mlp_sens,
        "dit_vs_mlp_perturb_ratio_h16": round(ratio_h16, 4),
        "gate_a": gate_a,
        "vae_figure": vae_figure_path,
    }

    out_json = f"{SPATIAL_DIR}/action_sensitivity_{encoder_name}_h{horizon}_s{seed}.json"
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[cf] wrote {out_json}")
    vol.commit()
    return result


def _entry(fn):
    if app is not None:
        return app.local_entrypoint()(fn)
    return fn


@_entry
def main(
    encoder: str = "vit_s16",
    seed: int = 0,
    horizon: int = 16,
    n_windows: int = N_WINDOWS_DEFAULT,
    steer_perturb: float = STEER_PERTURB,
):
    t0 = time.time()
    result = run_counterfactual.remote(encoder, seed, horizon, n_windows, steer_perturb)
    print(json.dumps(result, indent=2))
    out = Path(f"artifacts/full/action_sensitivity_{encoder}_h{horizon}_s{seed}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved {out} ({time.time() - t0:.0f}s)")