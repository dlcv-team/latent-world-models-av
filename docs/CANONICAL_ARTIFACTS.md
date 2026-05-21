# Canonical Artifacts

This document explains the contract that `configs/canonical.yaml` enforces
and what every team member needs to know to keep their PRs converging on a
shared set of numbers.

## Why this exists

Three workstreams running in parallel against the same dataset can silently
drift on splits, normalization, and seeds. Once that drift exists,
cross-encoder RMSE / DeltaCosSim comparisons become incomparable and the
report's headline claims fall apart. This contract makes the convergence
mechanical:

1. One config (`configs/canonical.yaml`) names every shared constant.
2. One script (`scripts/check_canonical_contract.py`) verifies nothing has
   drifted.
3. Two test files (`tests/test_canonical_contract.py`,
   `tests/test_reproduces_baselines.py`) are the merge gate in CI.

## What's pinned

| Pinned thing | Where | Owner |
|---|---|---|
| trainval-mirror subset (180 train / 20 val / 40 test scenes, seed 42, plus 80 P1/P2 scenes) | `configs/trainval_subset_manifest.json` (sha256 in canonical.yaml) | M1 |
| Action normalization (steer = clip(rad / 6.0, [-1, 1]); accel = clip(m_s2 / 10.0, [-1, 1])) | `dataset.normalization` in canonical.yaml | M1 / M2 |
| CAN-bus alignment tolerance (≤ 50 000 µs) and blacklist policy | `dataset.can_bus` | M2 |
| Probe architecture and hyperparams (Adam lr=1e-3, MSE, batch 256, 50 epochs, no tuning) | `probe` | M1 |
| BC baseline hyperparams (Adam lr=1e-3, early stop patience=10) | `bc_baseline` | M3 |
| Latent predictor architecture and hyperparams | `latent_predictor` | M1 |
| Encoder list and per-encoder framework / model id / projection / clip mode | `encoders` | M1 |
| Standardised target embedding dim (384) | `target_embedding_dim` | M1 |
| Bootstrap CI protocol (n=1000, seed 42) | `evaluation.bootstrap` | M2 |
| Paired-test correction policy (`ttest_rel` + Bonferroni; n_comparisons computed not asserted) | `evaluation.paired_tests` | M1 |
| Scenario bucket lists (P0 and P2-extended) | `evaluation.scenario_buckets*` | M2 / M3 |
| Figure DPI (300) and required caption strings | `figures` | M2 |

## What's NOT in the repo

* **Action labels CSV** (`data/raw/camfront_keyframe_actions.csv`, ~10 MB).
  Kept out of git. Resolution order:
    1. `$NUSCENES_ACTIONS_CSV` if set.
    2. `<repo_root>/data/raw/camfront_keyframe_actions.csv`.
  The contract check verifies sha256 matches the pinned value when the
  file is present and prints a warning when it isn't (so M2's data
  pipeline can fail loudly on first use).
* **nuScenes raw data**. Each member sets `NUSCENES_DATAROOT` locally.
* **Encoder embedding caches, probe checkpoints, figures**. Land under
  `outputs/` (gitignored).

## How to use it from module code

```python
from config import load_canonical, manifest_split

cfg = load_canonical()
test_scenes = manifest_split(cfg, "p0_test")  # 40 scene names

probe_lr = cfg.probe()["learning_rate"]
steer_div = cfg.normalization("steering")["divisor"]
```

**Do not** hardcode `0.001`, `42`, `6.0`, `180`, etc. anywhere in module
code. Reviewers will ask "what value of `configs/canonical.yaml` does this
come from?" and reject the PR if the answer is "it's hardcoded."

## How to change something pinned

1. Open a PR that bumps `version` in `canonical.yaml`.
2. Update this file with a one-line entry under "Change log".
3. Get explicit Slack approval from M1 (project lead) plus the workstream
   owner that the change affects most.
4. CI must still go green — if you change a sha256 you must also rerun any
   downstream test fixtures that rely on the old value.

## Change log

| Version | Date | Change |
|---|---|---|
| 1.0.0 | 2026-05-02 | Initial contract: trainval-mirror subset, seed 42, 5 encoders, probe / BC / LP hyperparams, 300 DPI figures, FR-08 VQ fallback policy. |
| 1.0.1 | 2026-05-14 | Action labels CSV regenerated with full 17-column schema (sha256: ff70d20f... → 18ba46c3..., byte-level diff only). Per-encoder envelopes in `tests/data/pilot_baselines.json` re-pinned to match `artifacts/pilot/canonical_closure/encoder_summary_with_ci_5enc.csv`; the `pending_revalidation` marker is dropped. `analysis/paired_tests.py` reproduces the pinned numbers within `tolerance.rmse_abs_atol`. New contract test `tests/test_reproduces_baselines.py::test_pilot_baselines_match_in_repo_canonical_closure` enforces the invariant going forward. |
| 1.0.2 | 2026-05-14 | VQ-VAE encoder: vendored VQGAN encoder from CompVis/taming-transformers (MIT) in `encoders/_vqgan_arch.py`; rewrote `encoders/vqvae.py` to load real Heidelberg checkpoint (`&dl=1` URL fix) instead of always falling back to DINOv2. Changed `model_id` from `vqgan_imagenet_f16_1024` to `vqgan_imagenet_f16_16384`, `framework` to `vendored_vqgan`. |
