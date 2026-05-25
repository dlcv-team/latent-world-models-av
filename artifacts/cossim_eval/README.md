# C4 CosSim evaluation artifacts

This directory holds the canonical per-horizon CosSim / DeltaCosSim
artifacts produced by `evaluation.latent_eval` (task C4) against the
exported `z_{hat,real}_{conditioned,unconditioned}.pt` tensors.

## Contents

| File | Producer | Consumer |
|---|---|---|
| `cossim_results.csv` | `python -m evaluation.latent_eval` (C4) | M2 figure pipeline; `analysis.delta_cossim_summary` (C5) |
| `cossim_results.json` | `python -m evaluation.latent_eval` (C4) | `analysis.delta_cossim_summary` (C5) for provenance |
| `delta_cossim_summary.md` | `python -m analysis.delta_cossim_summary` (C5) | Human readers / report writeup |

The CSV is the **shared deliverable for M2**. Its schema is fixed:

```
k,cossim_conditioned,cossim_unconditioned,delta_cossim
```

one row per prediction horizon. M2's figure scripts can `pd.read_csv` it
directly; the column order is exported as
`evaluation.latent_eval.CSV_COLUMNS` for cross-referencing.

The JSON mirrors the same numbers in a nested layout and additionally
carries a `metadata` block (`encoder`, `n_samples`, `horizon`, `z_dim`,
`generated_at_utc`, source paths) so any consumer can verify which
predictor run these numbers came from.

## How these get refreshed

When M1 re-exports `z_*.pt` (e.g., after retraining the latent
predictor), regenerate this directory with:

```bash
python -m evaluation.latent_eval \
    --z-hat-conditioned    outputs/z_hat/z_hat_conditioned.pt \
    --z-real-conditioned   outputs/z_hat/z_real_conditioned.pt \
    --z-hat-unconditioned  outputs/z_hat/z_hat_unconditioned.pt \
    --z-real-unconditioned outputs/z_hat/z_real_unconditioned.pt \
    --output-dir           artifacts/cossim_eval \
    --encoder              vjepa2_rep64
```

then commit the updated CSV / JSON. The downstream C5 summary is then
regenerated and re-committed with:

```bash
python -m analysis.delta_cossim_summary
cp outputs/analysis/delta_cossim_summary.md artifacts/cossim_eval/
```

M2's figures will re-derive automatically.

## Current run provenance

The committed numbers come from a single seed-0 run against the V-JEPA2
(`vjepa2_rep64`) latent predictor on the 5419-sequence test split. See
the `metadata` block in `cossim_results.json` for the full provenance
block (timestamp, sample count, source paths).
