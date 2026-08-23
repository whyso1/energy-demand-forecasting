# Audit: legacy notebook rebuild

Self-audit of the rebuild from `legacy/ml_code.ipynb` into this repository,
written for a second reviewer. Organized as: what was verified, what was
changed and why, what was tested, and what a reviewer should scrutinize
that I could not fully verify myself.

## 1. Scope

Input: `legacy/ml_code.ipynb` (audited, executed, with stored outputs),
`legacy/era5-s3-via-boto.ipynb` (ERA5 download script, unchanged, kept for
provenance), raw data in `~/Desktop/energy/` (ERA5 `.nc` files, EIA CSV).
Output: `src/`, `config/config.yaml`, `tests/`, `run_pipeline.py`,
`scripts/reproduce_legacy.py`, this repo's `README.md`.

## 2. Confirmed problems in the legacy notebook (verified against source, not taken on faith)

1. `test_train_data()` calls `sklearn.train_test_split(X, y, test_size=test_num, random_state=0)`
   with no `shuffle=False` -- default is `shuffle=True`, so train/test rows
   are randomly interleaved in time. Confirmed by reading the function body
   directly (`legacy/ml_code.ipynb`, cell 8).
2. `transform_data()` intersects weather and demand on identical timestamps
   with no shift anywhere in the pipeline before `test_train_data()`.
   Confirmed by reading cells 6, 12-17 -- there is no `.shift()` call on
   either series before they're joined.
3. No baseline model (persistence, seasonal-naive, or otherwise) appears
   anywhere in the notebook. Confirmed by reading all 27 cells.
4. `add_lag_mean()` is called only on `meteorology_data` (cell 14), never on
   `energy_data`. The demand series itself has no lag features.
5. Given (1)+(2), the notebook's own stored output --
   `XGBoost R²=0.9918` (cell 22, reproduced above in section 4) -- cannot be
   evidence of forecasting skill: the model was scored on rows it could have
   trained on temporal neighbors of, using weather read at the same instant
   as the demand it predicts.

## 3. Additional problems found during the rebuild (not in the original problem list)

6. **Demand series double-counting.** `preprocess_dataset_energy()` does
   `df.to_xarray().groupby(["period", "type"]).mean(dim="index")` with no
   filter on `respondent`. The EIA CSV has 76 respondent codes: individual
   balancing authorities (`CISO`, `ERCO`, `PJM`, ...), EIA's own regional
   rollups (`CAL`, `CENT`, `MIDW`, `NE`, `SE`, `SW`, `TEX`, ...), and `US48`
   (the national total) -- all averaged into one number per timestamp.
   Verified by scanning the CSV directly:
   `respondent` has 76 unique values including `US48`; the mean mixes
   national total, regional sub-totals, and individual BAs. This is fixed
   in `src/ingestion/eia.py` by filtering to one `respondent` (default
   `US48`) before aggregating.
7. **Gap-blind rolling/lag windows.** `add_lag_mean()` uses positional
   `.rolling(w)` / `.shift(l)`, which count *rows*, not hours. If the source
   data has a missing hour, "a 24-row window" silently stops meaning
   "24 hours." Verified this wasn't an issue for the data actually used
   (ERA5 has zero gaps in the CONUS-mean series; `US48`/`D` has zero missing
   hours in the 2025-01-01 to 2025-11-16 span -- checked directly, see
   section 6). Fixed anyway in `src/features/weather_features.py` and
   `src/features/target_features.py` by reindexing to a complete hourly
   grid first, since the assumption isn't guaranteed to hold on a different
   data pull.
8. **ERA5 publication latency.** The `.nc` files are final ERA5
   (`expver=0001`), not ERA5T. Final ERA5 publishes with roughly a 5-day
   lag in practice. This pipeline does not model that latency -- it treats
   ERA5 at `t-24h` as available at issuance time `t-24h`, which is not true
   of a real deployment. Documented as a limitation in `README.md`; not
   fixed, because fixing it means bringing in a different weather source
   (GFS/HRRR forecasts) or ERA5T, which is out of scope for this rebuild.
   **A reviewer should treat every reported metric as an upper bound
   conditional on weather being available at the lag used, not as a
   deployable forecast's expected accuracy.**

## 4. Legacy reproduction: numeric comparison

`scripts/reproduce_legacy.py` reruns the flawed methodology (random split,
no horizon, unfiltered-respondent demand) against the same raw `.nc`/`.csv`
files, to get a same-environment "before" number rather than relying on the
notebook's stored output from a different run/package versions.

| Model | Notebook's stored R² (original run) | This repo's reproduction |
|---|---|---|
| XGBoost | 0.9918 | 0.9816 |
| Random Forest | 0.9779 | 0.9634 |
| Linear Regression | 0.8184 | 0.7862 |

Directionally identical, not byte-identical. The gap is expected and,
as far as I can tell, benign: different `scikit-learn`/`xgboost` versions
between whenever the notebook was originally run and this environment
(`scikit-learn==1.6.1`, `xgboost==3.1.2`, `pandas==2.2.3` -- see
`requirements.txt`), and `train_test_split(random_state=0)` operating over
whatever the current `eia_load_data.csv` / `eta_2025_*.nc` pull contains,
which may not be byte-identical to whatever was on disk when the notebook
was last executed. **I did not independently verify the two runs used the
literal same input files** -- I only confirmed the code path and random
seed are identical and the results land in the same qualitative range
(R² > 0.96 for both tree models either way).

## 5. Corrected pipeline: what the leak-safety guarantee actually rests on

Two independent mechanisms, both unit-tested (`tests/test_no_leakage.py`):

- Weather features are computed on a strictly backward-looking basis
  (`rolling`/`shift`/`diff` never look forward), then the whole block is
  shifted forward by the horizon so a feature attributed to target time `t`
  is provably equal to what was computable at `t - horizon`
  (`test_weather_features_shifted_by_horizon_equal_source_values` asserts
  this equality directly, not just "no NaNs").
- Target (demand) lag/rolling features raise `ValueError` at construction
  time if any configured lag is shorter than the horizon
  (`compute_target_features`, tested by
  `test_target_lag_shorter_than_horizon_raises`), and a rolling-window test
  injects a synthetic spike at the exact target hour and asserts the
  feature for predicting that hour doesn't reflect it
  (`test_target_rolling_excludes_the_target_hour_itself`).

**What this does not prove:** these are unit tests against small synthetic
fixtures (`tests/conftest.py`, 30-day series) and one integration test
against a monkeypatched `build_dataset()` -- not a formal verification, and
not run against the full real dataset as part of the automated suite (the
full-data run is `run_pipeline.py`, executed manually, not under `pytest`).
A reviewer who wants stronger assurance should spot-check a handful of real
rows from `results/tables/corrected_pipeline_metrics.csv`'s underlying `X`
against the raw `.nc`/`.csv` files by hand, the same way the tests do it
synthetically.

## 6. Data checks performed (results asserted above, verifiable independently)

- `eia_load_data.csv`: 2,153,412 rows, 76 unique `respondent` values, 4
  `type` values (`D`=Demand, `DF`=Day-ahead demand forecast,
  `NG`=Net generation, `TI`=Total interchange) -- confirmed from the CSV's
  own `type-name` column, not inferred.
- `US48`/`D`: exactly one row per hourly `period`, 7,657 rows,
  2025-01-01 to 2025-11-16, zero missing hours, zero NaN values -- checked
  by building the expected complete hourly `DatetimeIndex` and diffing.
- `eta_2025_*.nc` (11 files, Jan-Nov 2025): variables `t2m`, `d2m`, `u10`,
  `v10`, `sp`, `msl`; `expver` uniformly `0001` (final ERA5, not
  preliminary ERA5T); hourly, no gaps checked in the CONUS-mean series
  (post spatial-mean `.dropna()` removed zero rows, implying no all-NaN
  timesteps in the CONUS box).
- `eia_day_ahead` (`DF`, `US48`) benchmark sanity check: MAPE ≈ 1.6% over
  the full year, ≈ 2.1% over the test window specifically -- both in the
  range published EIA day-ahead demand forecasts typically achieve, which
  is why I trust it as a real baseline rather than a data artifact (a
  MAPE far outside 1-5% would have suggested a unit mismatch or
  misalignment).
- Credential scan before any git operations:
  `grep -rIniE "aws_access_key|aws_secret|AKIA...|api_key|secret_key|password|token"`
  across all tracked file types, plus a manual check of
  `legacy/era5-s3-via-boto.ipynb`'s actual boto3 calls. Two apparent
  `AKIA`-pattern matches turned out to be substrings inside base64-encoded
  PNG image data in stored notebook outputs (matplotlib figures), not real
  keys; the notebook's actual S3 access uses
  `botocore.UNSIGNED` (anonymous, public bucket, no credentials). No real
  secrets found in any file intended for the repo.

## 7. What a reviewer should scrutinize that I flagged but did not resolve

- **Split boundaries are arbitrary fixed dates**
  (`config.yaml`: `train_end: 2025-08-15`, `val_end: 2025-09-30`), not
  chosen by any seasonal-balance criterion. The test window
  (Oct 1 - Nov 14) contains neither a summer AC-driven peak nor a winter
  heating-driven peak -- the reported metrics describe shoulder-season
  performance specifically, not year-round performance. Untested: how
  these models do on a peak-demand season.
- **`US48` + CONUS-mean weather is a coarse pairing.** Confirmed as an
  improvement over the original bug (section 3, item 6), but a single
  Texas heat wave, say, moves ERCOT demand far more than it moves the
  CONUS-mean temperature. The R²/MAPE reported here answers "how well does
  CONUS-mean weather forecast CONUS-total demand," not "how well does
  weather forecast demand" in general -- a single-BA version with
  BA-clipped weather would be a materially different (likely harder, or
  differently-hard) problem.
- **No test exercises `run_pipeline.py` or `scripts/reproduce_legacy.py`
  end-to-end.** All `pytest` coverage is at the module level
  (`src/features/*`, `src/models/*`, `src/evaluation/*`). The two
  orchestration scripts were run manually once each on real data (numbers
  in `results/tables/`) but aren't wired into CI or a repeatable test.
- **`holidays.US()` default** only encodes US federal holidays, not
  state-specific ones, and this is inherited unchanged from the legacy
  notebook -- reasonable for a national-aggregate target, would need
  revisiting for a single-state/BA target.
- I did not attempt to independently verify EIA's own DF publication
  timing/methodology beyond what the CSV's `type-name` column states and
  the MAPE sanity check in section 6 -- I'm relying on the sanity check's
  plausibility, not a primary-source EIA methodology document.

## 8. Not done

- No git commit or push has been made as of this audit -- pending the
  user's decision on repo visibility.
- No CI (GitHub Actions or otherwise) configured for this repo, unlike the
  sibling `HRRR project 2` repo which has a `tests.yml` workflow. Worth
  adding if this repo will see ongoing changes.
