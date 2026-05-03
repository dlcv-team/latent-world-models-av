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
   wherever VQ appears.

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
