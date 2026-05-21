# Encoder Checkpoint Manifest

## Cached Assets On VM

- DINOv2 checkpoint:
  - `~/.cache/torch/hub/checkpoints/dinov2_vits14_pretrain.pth` (`85M`)
- timm/Hugging Face cache directories:
  - `models--timm--vit_small_patch16_224.augreg_in21k_ft_in1k`
  - `models--timm--vit_base_patch32_clip_224.openai`

## Verified Loads

- `vit_small_patch16_224.augreg_in21k_ft_in1k` (timm): loaded and forward pass succeeded.
- `clip_vit_b32_openai` (open_clip): loaded and forward pass succeeded.
- `dinov2_vits14` (torch.hub): loaded and forward pass succeeded.

## Precomputed Reusable Features

- `mini_embeddings_200.npz` generated from `v1.0-mini` CAM_FRONT:
  - frames: `200`
  - vit feature shape: `[200, 384]`
  - clip feature shape: `[200, 512]`
  - dino feature shape: `[200, 384]`

## Not Yet Cached In This Run

- V-JEPA checkpoint (not pulled in this execution window).
- VQ-VAE / VQGAN checkpoint (not pulled in this execution window).

## Next Actions

- Add explicit checkpoint download script for V-JEPA and VQ checkpoints.
- Save checkpoint hashes and local paths into a machine-readable manifest (`.json`).
