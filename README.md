# latent-world-models-av

Benchmarks five pretrained visual encoders (ViT-S/16, DINOv2-S/14, CLIP
ViT-B/32, VQ-VAE, V-JEPA2 ViT-L) for latent-space autonomous driving world
models. By probing frozen embeddings to predict CAN-bus ego-actions on
nuScenes, we evaluate which encoders best preserve the spatial and temporal
dynamics required for action-conditioned planning.

## Project documents

The full PRD, EDD, Roles, and Implementation Plan live in the team's docs
repo. The implementation plan's per-member task tables are the ground truth
for what to work on; this README is just the developer entry point.

## Repo layout

```
configs/
  canonical.yaml               # the contract — see docs/CANONICAL_ARTIFACTS.md
  trainval_subset_manifest.json
config.py                      # canonical-config loader
encoders/                      # encoder wrappers (M1)
models/                        # ActionProbe, BC baseline, latent predictor (M1, M3)
data/                          # NuScenesFrameDataset, splits, scene captions (M2, M3)
evaluation/                    # RMSE, GradCAM, perturbation, CosSim (M2, M3)
training/                      # train_probe, train_bc, train_latent_pred (M1, M3)
analysis/                      # paired_tests, CIs (M1)
scripts/
  check_canonical_contract.py  # CLI guard (also runs in CI)
tests/
  test_canonical_contract.py
  test_reproduces_baselines.py
  data/pilot_baselines.json    # pinned reference numbers from M1's pilot run
docs/
  CANONICAL_ARTIFACTS.md
.github/workflows/ci.yml       # contract + reproducibility gate
```

## Setup

```bash
# Python 3.11 recommended (matches CI).
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Verify your local contract is intact:
python scripts/check_canonical_contract.py

# Run the floor-of-correctness tests:
PYTHONPATH=. pytest -q
```

## How to work in this repo

Three rules — CI enforces them, so a passing build means you've followed
them. There's no separate checklist to paste into PRs.

1. **Read shared constants from `configs/canonical.yaml`** via
   `config.load_canonical()`. Do not hardcode splits, seeds, normalization
   constants, or hyperparameters anywhere in module code.
2. **Every numeric result you produce ships with a CSV/JSON sidecar**
   under `outputs/`. Figure scripts read from those sidecars; nothing
   load-bearing lives only in a notebook.
3. **Figures are saved at `dpi=300`** and their captions name the
   trainval-mirror subset (180/20/40, seed 42) and the FR-08 VQ fallback
   policy if VQ uses fallback.

If a PR can pass `scripts/check_canonical_contract.py` and `pytest -q` on
a clean checkout, it satisfies rules 1–3 mechanically. Rule 2 is enforced
by the figure-side tests that land alongside Member 2's plotting code; if
you add a numeric result without a sidecar, those tests will fail.

See `docs/CANONICAL_ARTIFACTS.md` for the full contract and how to change
anything pinned.

## External data

The action labels CSV (~10 MB) is not committed. Resolution order:

1. `$NUSCENES_ACTIONS_CSV` if set,
2. `data/raw/camfront_keyframe_actions.csv` (gitignored).

`scripts/check_canonical_contract.py` verifies its sha256 when found.

The full nuScenes dataset (raw images, CAN bus expansion) is referenced via
`$NUSCENES_DATAROOT`; see `data/dataset.py` (forthcoming, M2 task B3) for
the loader.

## Dataset Splits

This project supports two split configurations:

### Smoke Splits (v1.0-mini)
Fast iteration and smoke testing during development.

- **smoke_train**: 8 scenes from nuScenes `mini_train`
- **smoke_val**: 1 scene from nuScenes `mini_val` (deterministic split, seed 42)
- **smoke_test**: 1 scene from nuScenes `mini_val` (deterministic split, seed 42)

### Benchmark Splits (v1.0-trainval)
Canonical splits from the project manifest (see `configs/canonical.yaml`).

- **p0_train**: 180 scenes (frozen subset for Phase 0 benchmark)
- **p0_val**: 20 scenes (frozen subset for Phase 0 validation)
- **p0_test**: 40 scenes (frozen subset for Phase 0 evaluation)
- **p1p2_scenes**: 80 additional scenes (reserved for future phases)

**CAN Filtering**: Pre-applied in canonical manifest. Scenes without CAN bus data are excluded.

**Verification**: All splits are SHA256-verified against the canonical manifest.

### Usage

```python
from data.splits import get_split_from_canonical
from data import NuScenesFrameDataset

# Get canonical benchmark splits
p0_train_scenes = get_split_from_canonical("p0_train")
p0_val_scenes = get_split_from_canonical("p0_val")

# Or use the dataset directly (canonical splits only)
train_dataset = NuScenesFrameDataset(split="p0_train", mode="single_frame")
```

**Note**: For smoke testing during development, v1.0-mini splits are available via internal API.

### Split Statistics

| Split | Scenes | Samples |
|-------|--------|---------|
| **Smoke (v1.0-mini)** | | |
| smoke_train | 8 | 323 |
| smoke_val | 1 | 41 |
| smoke_test | 1 | 40 |
| **Benchmark (v1.0-trainval)** | | |
| p0_train | 180 | ~18,000 |
| p0_val | 20 | ~2,000 |
| p0_test | 40 | ~4,000 |

*Note: Benchmark sample counts are approximate (depends on CAN alignment filtering).*

## Branching

| Branch | Owner | Scope |
|---|---|---|
| `main` | all | merge target; protected |
| `m1-encoders` | Member 1 | encoders, probe, latent predictor, statistics |
| `m2-data` | Member 2 | data pipeline, eval harness, GradCAM, figures |
| `m3-analysis` | Member 3 | BC baseline, CosSim eval, language conditioning |

Open feature branches off the appropriate member branch
(`m1/scaffold-canonical-contract`, `m2/dataset-single-frame`, etc.) and PR
into `main`.
