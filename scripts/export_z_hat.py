"""Export z_hat and z_real tensors for CosSim evaluation (A18).

Runs trained latent predictors (conditioned + unconditional) on the
held-out test set and saves ``(N_test, horizon, z_dim)`` tensors to disk.

Member 3 loads these via ``data.z_hat.load_z_hat`` / ``load_z_real``
in C4 (``evaluation/latent_eval.py``) to compute per-horizon CosSim and
DeltaCosSim.

Each variant is trained with its own adapter projection (when
native_dim != target_dim), so the conditioned and unconditional models
operate in **different 384-d subspaces**. We therefore export a separate
``z_real_<variant>.pt`` for each so that CosSim is always computed in
the correct subspace.  Additionally, a ``z_t.pt`` tensor is saved to
support a copy-baseline CosSim(z_t, z_real_{t+k}).

Output files
------------
- ``z_hat_conditioned.pt``    — conditioned predictor output
- ``z_hat_unconditioned.pt``  — unconditional predictor output
- ``z_real_conditioned.pt``   — adapter-projected real future latents (conditioned adapter)
- ``z_real_unconditioned.pt`` — adapter-projected real future latents (unconditional adapter)
- ``z_t_conditioned.pt``      — adapter-projected current latents (for copy baseline)

Action timing note
------------------
``action_t`` is the nearest CAN bus reading to frame t's timestamp
(typically within 5 ms at 100 Hz CAN rate), i.e. approximately the
"start of interval" action for the ``z_t -> z_{t+1}`` transition at
2 Hz keyframe rate.  See ``configs/canonical.yaml::dataset::can_bus``
for alignment policy details.

Usage
-----
    python scripts/export_z_hat.py --encoder vjepa2_rep64
    python scripts/export_z_hat.py --encoder vjepa2_rep64 --upload-hf
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from config import load_canonical
from data.temporal import TemporalEmbeddingDataset
from models.fourier_embed import FourierActionEmbedding
from models.latent_pred import LatentPredictor

# Native embedding dimensions per encoder.
# Matches training/train_latent_predictor.py::NATIVE_DIMS.
NATIVE_DIMS: dict[str, int] = {
    "vit_s16": 384,
    "dino_vits14": 384,
    "clip_b32": 512,
    "vq_track": 256,
    "vjepa2_rep64": 1024,
    "vjepa2_rep1": 1024,
}


def _load_checkpoint(
    ckpt_path: Path,
    target_dim: int,
) -> tuple[LatentPredictor, FourierActionEmbedding, nn.Module]:
    """Load predictor, fourier_embed, and adapter from a checkpoint.

    Returns
    -------
    tuple
        (predictor, fourier_embed, adapter) — all in eval mode on CPU.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)

    # Reconstruct adapter
    if ckpt.get("adapter_state_dict") is not None:
        native_dim = list(ckpt["adapter_state_dict"].values())[0].shape[1]
        adapter: nn.Module = nn.Linear(native_dim, target_dim, bias=False)
        adapter.load_state_dict(ckpt["adapter_state_dict"])
    else:
        adapter = nn.Identity()

    # Reconstruct fourier_embed + predictor from canonical config
    fourier_embed = FourierActionEmbedding.from_canonical()
    fourier_embed.load_state_dict(ckpt["fourier_embed_state_dict"])

    predictor = LatentPredictor.from_canonical()
    predictor.load_state_dict(ckpt["predictor_state_dict"])

    predictor.eval()
    fourier_embed.eval()
    adapter.eval()

    return predictor, fourier_embed, adapter


@torch.no_grad()
def _run_inference(
    predictor: LatentPredictor,
    fourier_embed: FourierActionEmbedding,
    adapter: nn.Module,
    loader: DataLoader,
    variant: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run inference on the full test set.

    Returns
    -------
    z_hat
        ``(N, horizon, z_dim)`` -- predicted future latents.
    z_real
        ``(N, horizon, z_dim)`` -- adapter-projected real future latents.
    z_t_all
        ``(N, z_dim)`` -- adapter-projected current-frame latents (for
        copy-baseline evaluation).
    """
    z_hat_parts: list[torch.Tensor] = []
    z_real_parts: list[torch.Tensor] = []
    z_t_parts: list[torch.Tensor] = []

    for batch in loader:
        z_t_native = batch["z_t"]
        action = batch["action"]
        z_future_native = batch["z_future"]

        # Adapter projection
        z_t = adapter(z_t_native)
        B, H, native_dim = z_future_native.shape
        z_future = adapter(
            z_future_native.reshape(B * H, native_dim)
        ).view(B, H, -1)

        # Action embedding
        a_embed = fourier_embed(action)
        if variant == "unconditioned":
            a_embed = torch.zeros_like(a_embed)

        z_hat = predictor(z_t, a_embed)

        z_hat_parts.append(z_hat)
        z_real_parts.append(z_future)
        z_t_parts.append(z_t)

    return (
        torch.cat(z_hat_parts, dim=0),
        torch.cat(z_real_parts, dim=0),
        torch.cat(z_t_parts, dim=0),
    )


def _print_delta_cossim(
    z_hat_cond: torch.Tensor,
    z_hat_uncond: torch.Tensor,
    z_real_cond: torch.Tensor,
    z_real_uncond: torch.Tensor,
    z_t: torch.Tensor,
) -> None:
    """Print per-horizon CosSim, DeltaCosSim, and copy baseline.

    Each variant is evaluated against its own adapter-projected z_real
    so CosSim is computed in the correct subspace.  The "copy baseline"
    row shows CosSim(z_t, z_real_{t+k}) -- the score a trivial identity
    predictor would achieve -- to contextualize the predictor's value.
    """
    horizon = z_real_cond.shape[1]
    print("\n  Sanity-check DeltaCosSim (authoritative computation is C4):")
    print(f"  {'k':>3}  {'CosSim_cond':>12}  {'CosSim_uncond':>14}  "
          f"{'Delta':>8}  {'CopyBaseline':>13}")
    for k in range(1, horizon + 1):
        cond = F.cosine_similarity(
            z_hat_cond[:, k - 1], z_real_cond[:, k - 1], dim=-1
        ).mean()
        uncond = F.cosine_similarity(
            z_hat_uncond[:, k - 1], z_real_uncond[:, k - 1], dim=-1
        ).mean()
        # Copy baseline: CosSim(z_t, z_real_{t+k}) using conditioned
        # adapter's z_real (both z_t and z_real come from the same adapter)
        copy = F.cosine_similarity(
            z_t, z_real_cond[:, k - 1], dim=-1
        ).mean()
        delta = cond - uncond
        print(
            f"  k={k}:  {cond:>12.6f}  {uncond:>14.6f}  "
            f"{delta:>8.6f}  {copy:>13.6f}"
        )
    print()


def _upload_to_hf(output_dir: Path, repo_id: str) -> None:
    """Upload z_hat / z_real .pt files to HuggingFace Hub."""
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError(
            "Set HF_TOKEN environment variable for HuggingFace upload."
        )

    from huggingface_hub import HfApi

    api = HfApi(token=token)
    files = sorted(output_dir.glob("*.pt"))
    print(f"[export_z_hat] Uploading {len(files)} files to {repo_id} ...")

    for f in files:
        size_mb = f.stat().st_size / 1e6
        print(f"  {f.name} ({size_mb:.1f} MB) ...", end=" ", flush=True)
        api.upload_file(
            path_or_fileobj=str(f),
            path_in_repo=f.name,
            repo_id=repo_id,
            repo_type="dataset",
        )
        print("done")

    print(f"[export_z_hat] Upload complete: https://huggingface.co/datasets/{repo_id}")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export z_hat and z_real tensors for CosSim evaluation."
    )
    parser.add_argument(
        "--encoder",
        default="vjepa2_rep64",
        choices=sorted(NATIVE_DIMS.keys()),
        help="Encoder whose embeddings to use (default: vjepa2_rep64).",
    )
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=Path("outputs/latent_predictors"),
        help="Root directory containing trained checkpoints.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/z_hat"),
        help="Output directory for .pt files (default: outputs/z_hat).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Training seed folder to load checkpoints from (default: 0).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Inference batch size (default: 256).",
    )
    parser.add_argument(
        "--upload-hf",
        action="store_true",
        help="Upload output .pt files to HuggingFace Hub after saving.",
    )
    parser.add_argument(
        "--hf-repo",
        default="surlac/lwm-av-embeddings",
        help="HuggingFace repo to upload to (default: surlac/lwm-av-embeddings).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)

    cfg = load_canonical()
    target_dim = cfg.target_embedding_dim
    lp_cfg = cfg.latent_predictor()
    horizon = int(lp_cfg["prediction_horizon"])

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load test set
    test_ds = TemporalEmbeddingDataset.from_encoder(
        args.encoder, split="test", horizon=horizon
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0
    )
    print(f"[export_z_hat] Test set: {len(test_ds)} sequences")

    # Each variant has its own adapter, so z_real differs per variant.
    z_hats: dict[str, torch.Tensor] = {}
    z_reals: dict[str, torch.Tensor] = {}
    z_t_cond: torch.Tensor | None = None

    for variant in ("conditioned", "unconditioned"):
        ckpt_path = (
            args.checkpoint_root
            / args.encoder
            / variant
            / f"seed_{args.seed}"
            / "checkpoint.pt"
        )
        if not ckpt_path.exists():
            print(f"[export_z_hat] ERROR: checkpoint not found: {ckpt_path}")
            return 1

        print(f"[export_z_hat] Loading {variant} checkpoint from {ckpt_path}")
        predictor, fourier_embed, adapter = _load_checkpoint(ckpt_path, target_dim)

        z_hat, z_real, z_t = _run_inference(
            predictor, fourier_embed, adapter, test_loader, variant
        )
        print(f"[export_z_hat] {variant}: z_hat={z_hat.shape}  z_real={z_real.shape}")

        # Save z_hat and per-variant z_real
        torch.save(z_hat, args.output_dir / f"z_hat_{variant}.pt")
        torch.save(z_real, args.output_dir / f"z_real_{variant}.pt")
        print(f"[export_z_hat] Saved z_hat_{variant}.pt, z_real_{variant}.pt")

        z_hats[variant] = z_hat
        z_reals[variant] = z_real

        # Keep z_t from the conditioned pass for the copy baseline
        if variant == "conditioned":
            z_t_cond = z_t
            torch.save(z_t, args.output_dir / "z_t_conditioned.pt")
            print(f"[export_z_hat] Saved z_t_conditioned.pt")

    assert z_t_cond is not None
    _print_delta_cossim(
        z_hats["conditioned"],
        z_hats["unconditioned"],
        z_reals["conditioned"],
        z_reals["unconditioned"],
        z_t_cond,
    )

    # Upload to HuggingFace if requested
    if args.upload_hf:
        _upload_to_hf(args.output_dir, args.hf_repo)

    print("[export_z_hat] Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
