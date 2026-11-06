# Statcast XYZ reconstruction

One configurable experiment pipeline. Every successful run replaces exactly four files in `results/`:

| File | Contents |
|---|---|
| `metrics.csv` | Long-format metrics, populations, sample sizes and applicable paired confidence intervals. Filter `evaluation`, `metric`, `method_id`, `season`, `contact_type`, and `pitch_group`; retain any category, contrast, cohort or stratum fields when comparing rows. |
| `players.parquet` | The existing aggregate batter-comparison cells: one row per batter/season/contact type/pitch group/method ID, with predicted and official rates, coverage, eligibility and league-relative tendencies. |
| `swings.parquet` | Exactly **974,886 unique tracked swings**. Pitch identifiers/characteristics, every existing method's estimates/classifications, historical eligibility flags, and available selected-whiff labels share this one table. |
| `report.md` | Comparisons, data-driven recommendations, limitations, verification status and the configuration used. |

## Reproduce the results

**Data availability:** Large input datasets in `data/` are untracked to keep the Git repository small. Results cannot yet be fully reproduced from a clone alone; the exact datasets must be provided separately.

From the repository root, with the existing datasets and frozen parameters in place:

```powershell
uv sync --locked                         # Set up the environment
uv run --no-sync python main.py run       # Reproduce the configured experiment
uv run --no-sync python main.py verify    # Check saved results against frozen benchmarks
```

`run` reads `experiment.json` and evaluates the methods offline on the same swing population, so classification accuracy and batter/pitch-group tendencies can be compared reproducibly. It verifies results before replacing the four files listed above; repeated runs overwrite them without accumulating outputs. Start with `results/report.md`, then use the CSV and Parquet files for detailed comparisons.

`verify` checks the preserved metrics, predictions, classifications, eligible populations and protected inputs. It prints a pass/fail result and creates no additional files. To run the regression tests: `uv run --no-sync python -m unittest discover -s tests -v`.

## Reading results

```python
import pandas as pd

metrics = pd.read_csv("results/metrics.csv", low_memory=False)
players = pd.read_parquet("results/players.parquet")
swings = pd.read_parquet("results/swings.parquet")

# Positive improvement means the method has LOWER error than the baseline.
comparisons = metrics[metrics.metric.eq("improvement")]
contrasts = comparisons[
    comparisons.target_metric.eq("within_batter_pitch_group_contrast_mae_pp")
]
```

`method_id` distinguishes axis, method, calibration stage and reference protocol. For example, `y_ev50_calibrated_frozen_2025`. Continuous swing columns end in `__value`; classifications end in `__classification`. X/Z units are inches; **Y is timing in milliseconds**, not a spatial coordinate. Missing inputs remain missing. Not every method produces a finite estimate on every swing.

Original author X/Z contact columns remain unavailable on whiffs. The explicitly named `published_whiff_regression` columns contain the available whiff formulas; all-swing aggregate scoring substitutes these as it did previously. Historical reference predictions are only populated for the originally evaluated route/population. Selected-tail predictions and labels are populated only for their observed benchmark sample; `savant_play_id` identifies that sample. No swing is duplicated for a protocol or selection list.

Player rates are proportions from 0 to 1. `predicted_low_rate`, `predicted_middle_rate`, `predicted_high_rate` and corresponding official columns mean:

| Axis | Low | Middle | High |
|---|---|---|---|
| X | tied up | centered | flail |
| Y | late | on time | early |
| Z | over | lined up | under |

`*_league_delta_pp` is a classification-rate deviation from that method's or official data's swing-weighted mean within the **same eligible batter/season/contact/pitch-group cohort**. Values are percentage points; ineligible player rows remain missing for these tendencies. `eligible` and coverage retain the original rules. Selected-tail and historical-protocol performance metrics retain their distinct populations in `metrics.csv`; their pitch estimates appear in the common swing table. These are not silently treated as full-population player leaderboards.

## Confidence intervals and decisions

For classification-rate MAE and within-batter pitch-group contrast MAE, comparisons use paired bootstrap resampling **clustered by batter**: 1,000 resamples, seed 180, 95% percentile intervals. Each resampled batter retains all of their paired observations. Classification comparisons preserve batter/group-cell weighting. Contrast comparisons average three category-error cells per batter/contrast; `ALL_CONTRASTS` pools cells across FB–BB, FB–CH and BB–CH.

`value` is the improvement when `metric == "improvement"`: **baseline error minus method error**, so positive is better. `baseline_method_id`, `baseline_kind`, `target_metric`, `ci_low`, `ci_high` and `excludes_zero` make the comparison explicit. Null-baseline comparisons are retained. Existing global-reference and calibration-gain intervals retain their original calculations, including the historical 5,000-resample reference intervals. Undefined/uncomputed intervals stay missing. These are descriptive intervals without multiple-comparison adjustment.

Published whiff regressions are labeled as available coefficient baselines, distinct from the original author's unavailable advanced trained models. Whiff accuracy intervals measure classification error on selected extreme-tail pitches, not aggregate contact-rate MAE. Contact validation uses aggregate official rates, not pitch-level contact truth. Raw contact-X MAE must not be compared directly with published multi-axis, all-swing MAE.

Prioritize category-rate agreement, batter rankings and within-batter pitch-group tendencies. The report derives recommendations from the existing scores and paired intervals; it never changes the configured methods or hardcodes a winner. Check the null baseline and coverage. Physical errors remain secondary diagnostics. A confidence interval spanning zero does not support a clear improvement.

## Configuration and preservation

`experiment.json` retains all methods and historical protocols. X includes author/global/batter/EV50 median/EV50 mean/historical-relative references. Y includes author/global/batter/EV50/Tango-time/historical-relative and pooled calibration. Z retains the original contact and published whiff formulas. Whiff proxies, development mean/median/majority baselines, B1/B2 X variants, frozen/same-season/prior-season references, and original author validation rules remain available.

2025 is development, 2024 backward transfer, and 2026 forward transfer. Both transfer seasons have already been examined. `existing` is a historical pooled/leaky sensitivity; same-season references are descriptive. Method selection changes displayed comparisons, never the fixed common-input population. The `xyz` experiment is required for the fixed swing-table contract.

`python main.py run --refit` recomputes the same development-only calibration in memory. It still writes only the four files and does not change frozen artifacts. `--config` and `--output` remain available; no export flags are needed.

Source datasets, `parameters/`, the original submodule and **`notebooks/scratch.ipynb`** remain unchanged. Source filenames used by scratch remain in place. `notebooks/analysis.ipynb` is a viewer of the four outputs. No notebook contains fitting or reconstruction logic.

Parquet metadata stores small schema mappings and configuration, not duplicate datasets. The mappings let verification reconstruct the old views in memory and check the existing frozen fingerprints and metric fixtures. `verify` reads the four files and prints its result; it creates no fifth file. Sample membership and classifications remain exact. NaN representation is canonicalized for in-memory hashing to match Parquet's on-disk representation; finite values are not rounded or changed.

## Data maintenance and archive

```powershell
python main.py collect --seasons 2024 2025 2026 --workers 12
python main.py match --seasons 2024 2025 2026
```

These explicit data-maintenance commands retain their existing caches and collection/matching audits under `data/savant_whiff_tails/`. They are separate from experiment runs and were not used to alter source data during the output change. Normal runs only read those inputs.

The earlier full repository backup and verified superseded-experiment archive remain outside the repository at `C:\Users\tyism\Projects\capstone-archive\`. `parameters/consolidation.json` records that earlier consolidation. It is preserved unchanged. Normal execution and verification do not depend on those external archives.
