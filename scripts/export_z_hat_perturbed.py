"""Export masked DiT predictions for perturbation analysis (B10 extension).

Generates DiT predictions on V-JEPA2 embeddings from perturbed input frames
(left lane, right lane, lead vehicle masks). Outputs .pt tensors for CosSim
evaluation comparing masked vs unmasked predictions.

Reuses:
- Masking functions from evaluation.perturbation
- DiT model definition from scripts/rollout_dit.py
- Checkpoint loading from scripts/export_z_hat.py

Usage:
    python scripts/export_z_hat_perturbed.py \\
        --encoder vjepa2_rep64 \\
        --seed 0 \\
        --perturbation mask_left_lane

    python scripts/export_z_hat_perturbed.py \\
        --encoder vjepa2_rep64 \\
        --seed 0 \\
        --perturbation mask_right_lane

    python scripts/export_z_hat_perturbed.py \\
        --encoder vjepa2_rep64 \\
        --seed 0 \\
        --perturbation mask_lead_vehicle

Output files (per perturbation type):
    outputs/z_hat_perturbed/{perturbation}/
        z_hat_conditioned_masked.pt      # DiT predictions with masked input
        z_hat_unconditioned_masked.pt    # DiT unconditional with masked input

Ground truth z_real tensors are reused from outputs/z_hat/ since they're
independent of input masking.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import load_canonical, manifest_split
from data.dataset import NuScenesFrameDataset
from nuscenes.nuscenes import NuScenes

# Import masking functions from perturbation pipeline
from evaluation.perturbation import apply_perturbation

# Inline DiT model definitions (same pattern as rollout_dit.py for portability)
# These MUST mirror configs/dit.yaml

DIT_CANONICAL = {
    "n_blocks": 4,
    "n_heads": 6,
    "z_dim": 384,
    "horizon": 4,
    "cond_dim": 384,
    "mlp_ratio": 4.0,
    "dropout": 0.0,
}

DIFFUSION_CANONICAL = {
    "n_train_steps": 1000,
}

FOURIER_CANONICAL = {
    "n_frequencies": 64,
    "base": 2.0,
    "out_dim": 384,
}

EVAL_CANONICAL = {
    "n_sample_steps": 50,
}

NATIVE_DIMS = {
    "vjepa2_rep64": 1024,
    "vjepa2_rep1": 1024,
}


# ---------------------------------------------------------------------------
# Inline DiT model definitions (from rollout_dit.py)
# ---------------------------------------------------------------------------


class TimestepEmbedding(nn.Module):
    """Sinusoidal timestep embedding + MLP projection."""

    def __init__(self, cond_dim=384):
        super().__init__()
        self.cond_dim = cond_dim
        self.mlp = nn.Sequential(
            nn.Linear(cond_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )

    def forward(self, timestep):
        half_dim = self.cond_dim // 2
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(half_dim, device=timestep.device, dtype=torch.float32)
            / half_dim
        )
        args = timestep.float().unsqueeze(-1) * freqs.unsqueeze(0)
        emb = torch.cat([args.sin(), args.cos()], dim=-1)
        return self.mlp(emb)


def _modulate(x, shift, scale):
    return x * (1.0 + scale) + shift


class DiTBlock(nn.Module):
    """Transformer block with adaLN-Zero conditioning."""

    def __init__(self, dim=384, cond_dim=384, n_heads=6, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm_attn = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=n_heads, dropout=dropout, batch_first=True
        )
        self.norm_mlp = nn.LayerNorm(dim, elementwise_affine=False)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden), nn.GELU(), nn.Linear(mlp_hidden, dim)
        )
        self.drop = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.adaln_linear = nn.Linear(cond_dim, 6 * dim)
        nn.init.zeros_(self.adaln_linear.weight)
        nn.init.zeros_(self.adaln_linear.bias)

    def forward(self, x, cond):
        mod = self.adaln_linear(cond).unsqueeze(1)
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = mod.chunk(6, dim=-1)
        h = _modulate(self.norm_attn(x), shift_a, scale_a)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + gate_a * self.drop(attn_out)
        h = _modulate(self.norm_mlp(x), shift_m, scale_m)
        x = x + gate_m * self.drop(self.mlp(h))
        return x


class LatentDiT(nn.Module):
    """DiT for latent future prediction (epsilon prediction)."""

    def __init__(
        self,
        z_dim=384,
        cond_dim=384,
        n_blocks=4,
        n_heads=6,
        horizon=4,
        mlp_ratio=4.0,
        dropout=0.0,
    ):
        super().__init__()
        self.z_dim = z_dim
        self.horizon = horizon
        self.input_proj = nn.Linear(z_dim, z_dim)
        self.timestep_embed = TimestepEmbedding(cond_dim)
        self.z_t_proj = nn.Linear(z_dim, cond_dim)
        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    dim=z_dim,
                    cond_dim=cond_dim,
                    n_heads=n_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
                for _ in range(n_blocks)
            ]
        )
        self.final_norm = nn.LayerNorm(z_dim, elementwise_affine=False)
        self.final_adaln = nn.Linear(cond_dim, 3 * z_dim)
        nn.init.zeros_(self.final_adaln.weight)
        nn.init.zeros_(self.final_adaln.bias)
        self.final_linear = nn.Linear(z_dim, z_dim)

    def forward(self, x_noisy, z_t, a_embed, timestep):
        cond = self.timestep_embed(timestep) + self.z_t_proj(z_t) + a_embed
        x = self.input_proj(x_noisy)
        for block in self.blocks:
            x = block(x, cond)
        mod = self.final_adaln(cond).unsqueeze(1)
        shift, scale, gate = mod.chunk(3, dim=-1)
        x = gate * self.final_linear(_modulate(self.final_norm(x), shift, scale))
        return x


class FourierActionEmbedding(nn.Module):
    """Fourier features for (steer, accel) -> dense embedding."""

    def __init__(self, action_dim=2, n_frequencies=64, base=2.0, out_dim=384):
        super().__init__()
        self.action_dim = action_dim
        self.n_frequencies = n_frequencies
        freqs = base ** torch.arange(n_frequencies, dtype=torch.float32) * torch.pi
        self.register_buffer("freqs", freqs)
        fourier_dim = action_dim * 2 * n_frequencies
        self.proj = nn.Sequential(
            nn.Linear(fourier_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, action):
        x = action.unsqueeze(-1) * self.freqs.unsqueeze(0).unsqueeze(0)
        x = torch.cat([x.sin(), x.cos()], dim=-1)
        x = x.reshape(x.shape[0], -1)
        return self.proj(x)


class CosineNoiseSchedule(nn.Module):
    """Cosine beta schedule for diffusion."""

    def __init__(self, n_steps=1000, s=0.008):
        super().__init__()
        self.n_steps = n_steps

        # Compute alphas per Nichol & Dhariwal 2021
        t = torch.arange(n_steps + 1, dtype=torch.float32) / n_steps
        alpha_bar = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
        alpha_bar = alpha_bar / alpha_bar[0]

        alphas = alpha_bar[1:] / alpha_bar[:-1]
        alphas = torch.clamp(alphas, max=0.9999)

        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alpha_bar[1:])
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alpha_bar[1:]))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alpha_bar[1:]))

    def add_noise(self, x_0, t, noise=None):
        """Add noise: x_noisy = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise."""
        if noise is None:
            noise = torch.randn_like(x_0)
        sqrt_alpha_bar = self.sqrt_alphas_cumprod[t].view(-1, 1, 1)
        sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)
        return sqrt_alpha_bar * x_0 + sqrt_one_minus_alpha_bar * noise, noise


class DDIMSampler:
    """Deterministic DDIM sampling."""

    def __init__(self, schedule: CosineNoiseSchedule, n_steps=50):
        self.schedule = schedule
        self.n_steps = n_steps
        # Subsample timesteps
        self.timesteps = torch.linspace(
            schedule.n_steps - 1, 0, n_steps, dtype=torch.long
        )

    @torch.no_grad()
    def sample(self, noise_pred_fn, shape, cond_kwargs, device):
        """Run DDIM reverse process from pure noise."""
        x = torch.randn(shape, device=device)

        for i, t in enumerate(self.timesteps):
            t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)
            noise_pred = noise_pred_fn(x, **cond_kwargs, timestep=t_batch)

            alpha_bar_t = self.schedule.alphas_cumprod[t]
            if i < len(self.timesteps) - 1:
                alpha_bar_prev = self.schedule.alphas_cumprod[self.timesteps[i + 1]]
            else:
                alpha_bar_prev = torch.tensor(1.0, device=device)

            # DDIM update
            pred_x0 = (x - torch.sqrt(1.0 - alpha_bar_t) * noise_pred) / torch.sqrt(
                alpha_bar_t
            )
            x = (
                torch.sqrt(alpha_bar_prev) * pred_x0
                + torch.sqrt(1.0 - alpha_bar_prev) * noise_pred
            )

        return x


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------


def _load_dit_checkpoint(
    ckpt_path: Path, encoder_name: str, target_dim: int, device: torch.device
) -> tuple[LatentDiT, FourierActionEmbedding, nn.Module, torch.Tensor, torch.Tensor]:
    """Load DiT, FourierEmbed, adapter, and normalization stats from checkpoint.

    Returns
    -------
    tuple
        (dit, fourier_embed, adapter, z_mean, z_std) — all in eval mode on device.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)

    # Check for required keys
    required_keys = ["dit_state_dict", "fourier_embed_state_dict", "z_mean", "z_std"]
    missing = [k for k in required_keys if k not in ckpt]
    if missing:
        raise RuntimeError(f"Checkpoint missing keys: {missing}")

    # Build adapter if needed
    native_dim = NATIVE_DIMS[encoder_name]
    needs_adapter = native_dim != target_dim

    if needs_adapter:
        if "adapter_state_dict" not in ckpt:
            raise RuntimeError(
                f"Encoder {encoder_name} requires adapter but checkpoint has no adapter_state_dict"
            )
        adapter = nn.Linear(native_dim, target_dim, bias=False).to(device)
        adapter.load_state_dict(ckpt["adapter_state_dict"])
        for p in adapter.parameters():
            p.requires_grad_(False)
    else:
        adapter = nn.Identity().to(device)

    # Build FourierActionEmbedding
    fourier_embed = FourierActionEmbedding(
        action_dim=2,
        n_frequencies=FOURIER_CANONICAL["n_frequencies"],
        base=FOURIER_CANONICAL["base"],
        out_dim=FOURIER_CANONICAL["out_dim"],
    ).to(device)
    fourier_embed.load_state_dict(ckpt["fourier_embed_state_dict"])

    # Build DiT
    dit = LatentDiT(
        z_dim=DIT_CANONICAL["z_dim"],
        cond_dim=DIT_CANONICAL["cond_dim"],
        n_blocks=DIT_CANONICAL["n_blocks"],
        n_heads=DIT_CANONICAL["n_heads"],
        horizon=DIT_CANONICAL["horizon"],
        mlp_ratio=DIT_CANONICAL["mlp_ratio"],
        dropout=DIT_CANONICAL["dropout"],
    ).to(device)
    dit.load_state_dict(ckpt["dit_state_dict"])

    dit.eval()
    fourier_embed.eval()

    z_mean = ckpt["z_mean"].to(device)
    z_std = ckpt["z_std"].to(device)

    return dit, fourier_embed, adapter, z_mean, z_std


# ---------------------------------------------------------------------------
# Main inference loop
# ---------------------------------------------------------------------------


@torch.no_grad()
def _run_masked_inference(
    dit: LatentDiT,
    fourier_embed: FourierActionEmbedding,
    adapter: nn.Module,
    encoder: nn.Module,
    sampler: DDIMSampler,
    loader: DataLoader,
    perturbation_type: str,
    nusc: NuScenes,
    variant: str,
    z_mean: torch.Tensor,
    z_std: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Run DiT inference on masked frames.

    Returns
    -------
    torch.Tensor
        Shape (N, horizon, z_dim) — DiT predictions on masked inputs.
    """
    z_hat_parts: list[torch.Tensor] = []
    skipped_count = 0

    for batch in tqdm(loader, desc=f"Masked inference ({perturbation_type}, {variant})"):
        # batch keys: "image_clip", "action", "sample_token"
        clip = batch["image_clip"].to(device)  # (B, 16, 3, 224, 224)
        action = batch["action"].to(device)  # (B, 2)
        sample_tokens = batch["sample_token"]  # list of str

        B = clip.shape[0]
        masked_clips = []

        for i in range(B):
            # Apply perturbation to each clip
            perturbed = apply_perturbation(
                clip[i], perturbation_type, nusc=nusc, sample_token=sample_tokens[i]
            )

            if perturbed is None:
                # No lead vehicle found — skip this sample
                skipped_count += 1
                masked_clips.append(None)
            else:
                masked_clips.append(perturbed)

        # Filter out None samples
        valid_indices = [i for i, m in enumerate(masked_clips) if m is not None]
        if len(valid_indices) == 0:
            continue

        valid_clips = torch.stack([masked_clips[i] for i in valid_indices]).to(device)
        valid_actions = action[valid_indices]

        # Embed masked frames
        with torch.no_grad():
            z_t_native = encoder(valid_clips)  # (B_valid, native_dim)

        # Adapter projection + normalize
        z_t = (adapter(z_t_native) - z_mean) / z_std  # (B_valid, 384)

        # Action embedding
        a_embed = fourier_embed(valid_actions)  # (B_valid, 384)
        if variant == "unconditioned":
            a_embed = torch.zeros_like(a_embed)

        # DDIM sampling
        z_hat = sampler.sample(
            noise_pred_fn=dit,
            shape=(len(valid_indices), DIT_CANONICAL["horizon"], DIT_CANONICAL["z_dim"]),
            cond_kwargs={"z_t": z_t, "a_embed": a_embed},
            device=device,
        )

        # Denormalize
        z_hat = z_hat * z_std + z_mean

        z_hat_parts.append(z_hat.cpu())

    if skipped_count > 0:
        print(f"  Skipped {skipped_count} samples (no lead vehicle found)")

    return torch.cat(z_hat_parts, dim=0)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export masked DiT predictions for perturbation analysis."
    )
    parser.add_argument(
        "--encoder",
        default="vjepa2_rep64",
        choices=["vjepa2_rep64", "vjepa2_rep1"],
        help="V-JEPA2 encoder variant (default: vjepa2_rep64).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="DiT checkpoint seed (default: 0).",
    )
    parser.add_argument(
        "--perturbation",
        required=True,
        choices=["mask_left_lane", "mask_right_lane", "mask_lead_vehicle"],
        help="Perturbation type to apply.",
    )
    parser.add_argument(
        "--dit-root",
        type=Path,
        default=Path("outputs/dits"),
        help="Root directory containing DiT checkpoints (default: outputs/dits).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/z_hat_perturbed"),
        help="Output root directory (default: outputs/z_hat_perturbed).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Inference batch size (default: 32).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)

    cfg = load_canonical()
    target_dim = cfg.target_embedding_dim
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[export_z_hat_perturbed] Encoder: {args.encoder}")
    print(f"[export_z_hat_perturbed] Perturbation: {args.perturbation}")
    print(f"[export_z_hat_perturbed] Seed: {args.seed}")
    print(f"[export_z_hat_perturbed] Device: {device}")

    # Output directory
    output_dir = args.output_root / args.perturbation
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load encoder
    print(f"[export_z_hat_perturbed] Loading encoder...")
    if args.encoder == "vjepa2_rep64":
        from encoders.vjepa2 import VJEPAEncoder

        encoder = VJEPAEncoder(variant="rep64", pretrained=True).to(device).eval()
    elif args.encoder == "vjepa2_rep1":
        from encoders.vjepa2 import VJEPAEncoder

        encoder = VJEPAEncoder(variant="rep1", pretrained=True).to(device).eval()
    else:
        raise ValueError(f"Unsupported encoder: {args.encoder}")

    # Load test set (V-JEPA2 requires multi-frame mode)
    test_scenes = manifest_split(cfg, "p0_test")
    test_ds = NuScenesFrameDataset(
        scene_names=test_scenes, mode="multi_frame", action_source="can_bus"
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0
    )
    print(f"[export_z_hat_perturbed] Test set: {len(test_ds)} samples")

    # Load NuScenes for masking
    nusc = NuScenes(
        version="v1.0-trainval", dataroot=cfg.nuscenes_dataroot, verbose=False
    )

    # Build DDIM sampler
    schedule = CosineNoiseSchedule(n_steps=DIFFUSION_CANONICAL["n_train_steps"]).to(
        device
    )
    sampler = DDIMSampler(schedule, n_steps=EVAL_CANONICAL["n_sample_steps"])

    # Process both variants
    for variant in ("conditioned", "unconditioned"):
        ckpt_path = (
            args.dit_root / f"{args.encoder}_s{args.seed}_{variant}" / "checkpoint.pt"
        )
        if not ckpt_path.exists():
            print(f"[export_z_hat_perturbed] ERROR: checkpoint not found: {ckpt_path}")
            return 1

        print(f"\n[export_z_hat_perturbed] Loading {variant} DiT checkpoint...")
        dit, fourier_embed, adapter, z_mean, z_std = _load_dit_checkpoint(
            ckpt_path, args.encoder, target_dim, device
        )

        print(f"[export_z_hat_perturbed] Running {variant} inference...")
        z_hat = _run_masked_inference(
            dit,
            fourier_embed,
            adapter,
            encoder,
            sampler,
            test_loader,
            args.perturbation,
            nusc,
            variant,
            z_mean,
            z_std,
            device,
        )

        # Save
        out_path = output_dir / f"z_hat_{variant}_masked.pt"
        torch.save(z_hat, out_path)
        print(
            f"[export_z_hat_perturbed] Saved {out_path.name}: shape={tuple(z_hat.shape)}"
        )

    print("\n[export_z_hat_perturbed] Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
