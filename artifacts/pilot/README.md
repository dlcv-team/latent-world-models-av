# Pilot artifacts

Pre-computed outputs from a GCP pilot run that the analysis, figure,
and latent-predictor pipelines all consume. The pilot followed the
exact evaluation protocol pinned in `configs/canonical.yaml` (5 CV
folds × 3 seeds × 40 test scenes), so the numbers here are the
canonical-closure numbers, not a loose preview.

Anyone with a fresh clone can run:

```bash
PYTHONPATH=. python scripts/adopt_pilot_artifacts.py
```

to transform this directory into the canonical
`outputs/probes/<encoder>/` sidecar layout that downstream code reads.

## Layout

| Subdir | Contents |
|---|---|
| `canonical_closure/` | Final 5-encoder summary CSVs (per-encoder mean RMSE, bootstrap CIs, paired t-tests with Bonferroni correction, BC baseline row, perturbation summary) + a JSON manifest listing them. Consumed by the paired-t-test analysis and the publication-ready figure scripts. |
| `per_scene/` | Per-scene RMSE, fold and scenario breakdowns, per-encoder seed metrics, and the train+eval phase report. Consumed by the paired-t-test analysis and the test-reproducibility checks. |
| `perturbation/` | Per-encoder perturbation RMSE, per-scene perturbation breakdown, and a summary. Consumed by the perturbation-analysis figure. |
| `retry_reports/` | `vq_retry_report.json` documents why the VQ-VAE wrapper falls back to DINOv2 embeddings (no loadable pretrained VQ checkpoint); `vjepa2_retry_report.json` documents the successful Hugging Face `transformers` load path for V-JEPA2. Reviewers use these to verify the fallback decision. |
| `embeddings/` | `camfront_keyframes_all_merged.npz` holds precomputed 384-d embeddings for all 5 encoders on the 3,500 keyframes in the trainval-mirror subset (keys: `cam_tokens`, `scene_names`, `timestamps_us`, `image_paths`, `vit_s16`, `clip_b32`, `dino_vits14`, `vq_track`, `vjepa2_rep64`, `vjepa2_rep1`). Plus per-temporal-stride V-JEPA caches for the multi-frame ablation. Consumed by latent-predictor training, CosSim evaluation, and attribution overlay pipelines that need cached embeddings. |
| `figures/` | Reference figure PDFs/PNGs from the pilot's figure stage. **Not final.** The figure scripts will regenerate these from `paired_tests`, `encoder_summary_with_ci`, and per-scenario RMSE with the up-to-date caption and Bonferroni-bracket logic. Use these as visual templates while iterating. |
| `smoke/` | v1.0-mini smoke run: encoder forward check, CAN-bus alignment check, environment report. Reference for the data pipeline's smoke tests. |

## Provenance notes

* **Action-labels SHA mismatch is expected.** Each `provenance.json`
  the adopt script writes records the pilot CSV SHA (`ff70d20f…`),
  which differs from the current
  `configs/canonical.yaml::dataset.action_labels.sha256` value
  (`18ba46c3…`). The newer canonical CSV is a byte-for-byte rotation
  of the pilot CSV (same scientific content, different column order
  and float formatting); the rotation is documented in
  `docs/CANONICAL_ARTIFACTS.md` and tracked by
  `tests/data/pilot_baselines.json`'s `status: pending_revalidation`
  marker, which is resolved once fresh canonical numbers land.

* **VQ-VAE is a documented fallback.** The `vq_retry_report.json`
  records the failed VQGAN checkpoint loads; the wrapper substitutes
  DINOv2-S/14 embeddings instead. The adopt script tags `vq_track`'s
  `provenance.json::fallback_caveat` accordingly so figure captions
  pick the caveat string up from data, not from hardcoded literals.

* **`vjepa2_rep1` rows in `per_scene/per_scene_rmse.csv` are the
  1-frame V-JEPA2 ablation.** The adopt script filters them out by
  default; the 5-encoder canonical row uses `vjepa2_rep64` (the
  64-frame V-JEPA2 path).

## Regenerating

These files were produced by the GCP pilot; the adopt script transforms
them into runtime sidecars under `outputs/probes/`. To re-run from
scratch (e.g., after an encoder wrapper change), use
`training/train_probe.py`. Any change to a file under this directory
requires the joint sign-off documented in
`docs/CANONICAL_ARTIFACTS.md`.
