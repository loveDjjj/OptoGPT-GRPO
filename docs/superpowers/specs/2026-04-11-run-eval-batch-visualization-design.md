# run_eval Batch Visualization Design

## Goal
- Extend `our_work/pretrain/scripts/run_eval.py` from a smoke JSON dumper into a batch evaluation entrypoint that produces normalized artifacts for later offline analysis and server runs.
- Keep evaluation independent from the root reinforcement-learning code and operate only on `our_work` checkpoints, shard datasets, and migrated physics utilities.

## Chosen Approach
- Keep a single `run_eval.py` entrypoint.
- One execution should both evaluate samples and materialize analysis artifacts.
- Default outputs are `summary.json + results.jsonl + PNG plots`.

## Output Layout
- Each run writes into:
  - `our_work/pretrain/outputs/<run_name>/eval_runs/<timestamp>/`
- Required files:
  - `summary.json`
  - `results.jsonl`
  - `plots/rmse_hist.png`
  - `plots/mae_hist.png`
  - `plots/target_layer_count_hist.png`
  - `plots/predicted_layer_count_hist.png`
  - `plots/per_layer_rmse_bar.png`
  - `samples/worst-<rank>-<sample_id>.png`
  - `samples/random-<rank>-<sample_id>.png`

## Data Products

### summary.json
- `metadata`
  - checkpoint path
  - dataset path
  - split
  - max samples
  - max new tokens
  - wavelength range
  - num points
  - incident angle
  - polarization
  - tolerance
  - timestamp
- `global_metrics`
  - `sample_count`
  - `valid_generation_count`
  - `valid_generation_rate`
  - `exact_match_count`
  - `exact_match_rate`
  - `mean_spectrum_rmse`
  - `mean_spectrum_mae`
- `per_target_layer_count`
  - one object per target layer count
  - fields:
    - `sample_count`
    - `valid_generation_count`
    - `valid_generation_rate`
    - `exact_match_count`
    - `exact_match_rate`
    - `mean_spectrum_rmse`
    - `mean_spectrum_mae`
- `artifacts`
  - relative paths for plots and sample figures
- `skipped_artifacts`
  - map from artifact name to skip reason when a plot is intentionally not generated

### results.jsonl
- One JSON object per evaluated sample.
- Required fields:
  - `sample_id`
  - `target_layer_count`
  - `prediction_layer_count`
  - `target_tokens`
  - `predicted_tokens`
  - `token_exact_match`
  - `generated_valid`
  - `spectrum_rmse`
  - `spectrum_mae`
- Optional fields added for visualization support:
  - `target_spectrum_rt`
  - `predicted_spectrum_rt`
  - `sample_figure_path`
  - `selection_bucket`

## Plotting Rules
- Distribution plots only use rows with numeric metrics.
- `rmse_hist.png` uses valid `spectrum_rmse`.
- `mae_hist.png` uses valid `spectrum_mae`.
- `target_layer_count_hist.png` uses all rows.
- `predicted_layer_count_hist.png` uses all rows.
- `per_layer_rmse_bar.png` uses grouped valid rows by `target_layer_count`.
- Sample figures are selected as:
  - worst `N` valid samples by `spectrum_rmse`
  - random `N` additional valid samples from the remaining pool
- Each sample figure overlays:
  - `target_R`
  - `pred_R`
  - `target_T`
  - `pred_T`
- Figure title includes:
  - sample id
  - target layer count
  - predicted layer count
  - exact match flag
  - RMSE

## Error Handling
- Invalid generated structures and TMM failures still emit a row in `results.jsonl`.
- Invalid rows count toward `sample_count` but do not contribute to RMSE/MAE means or histogram inputs.
- If there are not enough valid rows for a plot category, skip the artifact and record the reason in `summary.json`.
- Path resolution must continue supporting worktree-relative dataset and database paths.

## CLI Additions
- `--output-dir`
  - optional override for run artifact root
- `--worst-sample-plots`
  - default count for worst-case spectrum figures
- `--random-sample-plots`
  - default count for random spectrum figures
- `--disable-plots`
  - optional switch for headless environments when only JSON artifacts are needed

## Testing
- Add tests for:
  - JSONL writing
  - summary aggregation with per-layer metrics
  - plot generation when valid rows exist
  - artifact skipping when no valid rows exist
  - sample selection policy for worst + random figures
  - end-to-end smoke evaluation writing a run directory with required artifacts
