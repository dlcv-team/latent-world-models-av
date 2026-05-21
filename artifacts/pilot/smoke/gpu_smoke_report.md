# GPU Smoke Report

## Checks

- `nvidia-smi`: passed
- `torch.cuda.is_available()`: `true`
- `torch.cuda.device_count()`: `1`
- Device: `NVIDIA L4`

## Encoder Forward Smoke (single image, batch=1)

- `vit_small_patch16_224`: output `[1, 384]`, forward `0.2341s`, peak mem `101,385,728 bytes` (~96.7 MiB)
- `clip_vit_b32_openai`: output `[1, 512]`, forward `0.0163s`, peak mem `717,263,360 bytes` (~684.0 MiB)
- `dinov2_vits14`: output `[1, 384]`, forward `0.0172s`, peak mem `804,158,976 bytes` (~766.9 MiB)

## Notes

- No runtime errors reported in `encoder_smoke_report.json`.
- DINOv2 emitted warnings about missing xFormers (expected in this setup).
