# Audit: legacy notebook rebuild

Self-audit of the rebuild from `legacy/ml_code.ipynb` into this repository,
written for a second reviewer. Current as of commit `be5611d` (December 2025
data backfill). Organized as: what was verified, what was changed and why,
what was tested, what's still open, and what a reviewer should scrutinize
that I could not fully verify myself.

## 1. Scope

Input: `legacy/ml_code.ipynb` (audited, executed, with stored outputs),
`legacy/era5-s3-via-boto.ipynb` (ERA5 download script -- turned out to be a
stale example, see section 9), raw data in `~/Desktop/energy/` (ERA5T `.nc`
files, EIA CSV). Output: `src/`, `config/config.yaml`, `tests/`,
`run_pipeline.py`, `scripts/`, `.github/workflows/`, this repo's `README.md`.

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
   `XGBoost R²=0.9918` (cell 22, reproduced in section 4) -- cannot be
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
   Verified by scanning the CSV directly: `respondent` has 76 unique values
   including `US48`. Fixed in `src/ingestion/eia.py` by filtering to one
   `respondent` (default `US48`) before aggregating.
7. **Gap-blind rolling/lag windows.** `add_lag_mean()` uses positional
   `.rolling(w)` / `.shift(l)`, which count *rows*, not hours -- silently
   wrong if the source data has a missing hour. This turned out to affect
   two code paths, found at two different times:
   - The weather/target feature builders (`src/features/weather_features.py`,
     `src/features/target_features.py`), fixed in the initial rebuild by
     reindexing to a complete hourly grid before any shift/rolling.
   - The benchmark builders (persistence, seasonal-naive) in
     `src/features/dataset.py`, missed in the initial rebuild -- flagged by
     an independent follow-up audit, fixed in commit `29551b5` by extracting
     `compute_benchmark_features()` with the same grid-reindex guard. The
     fix's test (`test_persistence_and_seasonal_naive_use_exact_hour_offset_despite_gap`)
     was verified empirically against the pre-fix logic: for a target time
     whose lookback window straddles a dropped hour, the old code returned
     185.0 where the correct answer is 186.0 -- confirmed by running both
     versions side by side, not just asserted.
   In both cases this was benign for the actual data used (`US48`/`D` and
   the CONUS-mean ERA5T series both have zero missing hours in the current
   date range -- see section 6), but the code wasn't actually enforcing that
   property before the fix; it just happened to hold.
8. **ERA5 provenance was wrong in an earlier version of this document.**
   I originally wrote that the `.nc` files were confirmed-final ERA5
   (based on `expver=0001` in the file metadata). That was an inference,
   not a verification, and it was wrong: the actual acquisition script
   (`~/Desktop/energy/save_data.ipynb`, discovered later, not the
   `legacy/era5-s3-via-boto.ipynb` file that's actually in this repo -- that
   one is a stale, unused AWS Open Data example, see section 9) explicitly
   requests **ERA5T** (Copernicus's preliminary/near-real-time product) via
   the CDS API. `expver=0001` does not reliably distinguish ERA5T from final
   ERA5 for this CDS endpoint. Corrected in `README.md`; the practical
   consequence is the same directionally (treat weather as not-fully-final,
   possibly revised later) but the specific mechanism/lag figures I'd stated
   were guesses dressed as facts. **Lesson for the next reviewer: don't
   trust my characterization of a data source's provenance without checking
   whatever script actually produced the file, if one is available.**

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

Directionally identical, not byte-identical -- plausibly different
`scikit-learn`/`xgboost` versions and/or a non-identical underlying data
pull between the original notebook run and this environment. **I did not
independently verify the two runs used the literal same input files.**

## 5. Corrected pipeline: what the leak-safety guarantee actually rests on

Two independent mechanisms, both unit-tested (`tests/test_no_leakage.py`):

- Weather features are computed on a strictly backward-looking basis, then
  shifted forward by the horizon so a feature attributed to target time `t`
  is provably equal to what was computable at `t - horizon`
  (`test_weather_features_shifted_by_horizon_equal_source_values` asserts
  this equality directly).
- Target (demand) lag/rolling features raise `ValueError` at construction
  time if any configured lag is shorter than the horizon, and a
  rolling-window test injects a synthetic spike at the exact target hour and
  asserts the feature for predicting that hour doesn't reflect it.
- (Added in the follow-up pass) Benchmark features go through the same
  grid-reindex guard as the ML feature path -- see section 3, item 7.

**What this does not prove:** these are unit tests against small synthetic
fixtures and monkeypatched integration tests, not a formal verification, and
not run against the full real dataset under `pytest` (that's `run_pipeline.py`,
executed manually). A reviewer who wants stronger assurance should spot-check
real rows from `results/tables/corrected_pipeline_metrics.csv`'s underlying
`X` against the raw files by hand.

## 6. Data checks performed (verifiable independently)

- `eia_load_data.csv`: 2,153,412 rows, 76 unique `respondent` values, 4
  `type` values (`D`=Demand, `DF`=Day-ahead demand forecast,
  `NG`=Net generation, `TI`=Total interchange) -- confirmed from the CSV's
  own `type-name` column.
- `US48`/`D`: exactly one row per hourly `period`, zero missing hours, zero
  NaN values, originally 2025-01-01 to 2025-11-16 -- checked by building the
  expected complete hourly `DatetimeIndex` and diffing.
- `eta_2025_*.nc` (now 12 files, Jan-Dec 2025): variables `t2m`, `d2m`,
  `u10`, `v10`, `sp`, `msl`; hourly.
- `eia_day_ahead` (`DF`, `US48`) benchmark sanity check: MAPE ≈ 1.6-2.1%,
  in the range published EIA day-ahead demand forecasts typically achieve.
- Credential scan before any git operations (both the original commit and
  the December backfill): `grep` for AWS/API key/token/secret patterns
  across all tracked file types. Two apparent `AKIA`-pattern matches in
  `legacy/era5-s3-via-boto.ipynb` turned out to be substrings inside
  base64-encoded PNG image data in stored notebook outputs, not real keys.
  No secrets found in any file that was actually committed.

## 7. December 2025 backfill (commits `5515f8e`, `be5611d`)

User asked to extend the data to a full calendar year while away, "running
on GitHub" rather than locally. What happened, for the record:

- **Discovered the real acquisition scripts.** Neither of the two notebooks
  already in `legacy/` was actually how the data was built.
  `~/Desktop/energy/save_data.ipynb` (not part of this repo) has the real
  EIA pull (`requests` against `api.eia.gov/v2/...`) and the real ERA5T pull
  (`cdsapi` against Copernicus CDS) -- see section 3, item 8.
- **A live EIA API key was sitting in plaintext** in `save_data.ipynb`. It
  got echoed once into this session's tool output while I was searching for
  it (low-stakes -- free-tier EIA key, no billing/infra attached -- but
  disclosed to the user at the time). It was never committed to git.
  **Resolution (pre-publish review):** attempted rotation before making the
  repo public. EIA's registration system turned out not to support
  self-service key rotation at all -- registering the same email again
  returns "already registered," and "Forgot My API Key" only re-sends the
  existing key, it doesn't reissue a new one. Given that constraint, and
  that the exposure was confined to this session's own tool output (never
  committed, never posted publicly) on a free-tier key with no billing
  attached, the user's decision was to leave the existing key in place
  rather than register a new key under a different email. Re-confirmed via
  `grep` that no key value is committed anywhere in the repo (clean).
- **Credential handling for the automated pull:** with the user's explicit
  go-ahead, the existing EIA key and the existing `~/.cdsapirc` (Copernicus
  credentials) were added as encrypted GitHub Actions repo secrets
  (`EIA_API_KEY`, `CDSAPIRC`) via `gh secret set`, piping values directly
  from a Python subprocess into the CLI's stdin so neither value was ever
  printed to any tool output or file. Reviewer note: I did not myself
  re-verify after the fact that these secret values are truly inaccessible
  from workflow logs beyond GitHub's own masking guarantee -- that's a
  platform-level guarantee I'm trusting, not something I independently
  tested.
- **New scripts** (`scripts/pull_eia_month.py`, `scripts/pull_era5t_month.py`)
  mirror the exact request shape of the original working scripts (same EIA
  endpoint/facets, same CDS variable list and CONUS bounding box), scoped
  down to just `US48`/`D`+`DF` for the EIA side (no need to re-pull all 76
  respondents for one month). `pull_eia_month.py` was smoke-tested locally
  against the real API for a few hours of December before being trusted to
  run unattended.
- **New workflow** (`.github/workflows/pull_december_2025.yml`,
  `workflow_dispatch`-triggered, not scheduled -- a one-off backfill, not a
  standing job). Ran on GitHub-hosted runners, not the user's machine.
  EIA job: ~19 seconds, committed `data/pulled/eia_us48_202512.csv` (1,488
  rows: 744 hours x 2 types) directly to `main`. ERA5T job: ~7 minutes
  (Copernicus's queue was fast this run; this is not a reliable number --
  CDS queue times are known to vary from minutes to hours), produced
  `eta_2025_12.nc` as a workflow artifact (too large, ~200MB, to commit to
  git directly).
- **Local integration:** downloaded the artifact, validated it (`xarray`
  open, checked variable names and the full December `valid_time` range),
  moved it to `~/Desktop/energy/eta_2025_12.nc`, symlinked it into
  `data/raw/` the same way as the other 11 months.
- **Verified inert by design, not by luck:** re-ran `pytest` (17/17 pass)
  and `run_pipeline.py` with the December weather file present. Output was
  bit-for-bit identical to before, because `build_dataset()`'s
  `common_index` intersects weather/target/demand indices -- December
  weather with no corresponding December demand simply drops out of the
  intersection. This was checked, not assumed.

**What is explicitly NOT done yet, by design:** December's EIA data lives in
a separate file (`data/pulled/eia_us48_202512.csv`), not merged into the
`eia_load_data.csv` that `config/config.yaml` actually points at. Merging it
and deciding whether/how to move `split.val_end`/the test window to actually
exercise a winter month changes what the reported metrics mean --
deliberately left for the user to decide rather than done unilaterally.

## 8. What a reviewer should scrutinize that I flagged but did not resolve

- **Split boundaries are arbitrary fixed dates**, and the current test
  window (Oct 1 - Nov 14) contains neither a summer AC peak nor a winter
  heating peak. December data now exists to partially address this but
  isn't wired in yet (section 7).
- **`val` split is unused** (`chronological_split()` returns it,
  `run_pipeline.py` doesn't read it) -- disclosed in README rather than
  built out, per explicit user direction to keep that pass scoped.
- **`US48` + CONUS-mean weather is a coarse pairing** -- fixes the
  double-counting bug (section 3, item 6) but doesn't address that a single
  region's weather event can move that region's demand far more than it
  moves the CONUS mean.
- **No test exercises `run_pipeline.py`, `scripts/reproduce_legacy.py`, or
  the new pull scripts end-to-end under `pytest`.** All are run manually
  and checked by hand.
- **`holidays.US()`** only encodes US federal holidays, inherited unchanged
  from the legacy notebook.
- I have not independently verified EIA's own DF publication
  timing/methodology, or Copernicus's exact ERA5T-to-final-ERA5 revision
  policy, beyond what's stated in each provider's own general documentation
  as I understand it -- neither is a primary-source citation I've checked
  against the actual current EIA/Copernicus docs in this session.
- **GitHub Actions workflow has not been exercised against failure modes**
  -- e.g. what happens if the CDS request errors out after the job's
  350-minute timeout, or if EIA's API schema changes. The retry logic
  mirrors the original notebook's, but neither script has an automated test.

## 9. Correction to a previous version of this document

An earlier version of this file (and `README.md`) stated
`legacy/era5-s3-via-boto.ipynb` was "the ERA5 download script." That's
wrong: it's a stale, generic AWS Open Data example notebook (dates in its
own examples are 2017-2018) pointed at the `era5-pds` S3 bucket, which uses
a completely different file layout (per-variable files, 0-360° longitude)
than the actual `eta_2025_*.nc` files (which are cfgrib-converted GRIB
output from the CDS API, per each file's own `history` attribute). It was
never actually run to produce any data in this project. The real
acquisition script is `~/Desktop/energy/save_data.ipynb` (see section 7),
which is not part of this repo. `legacy/era5-s3-via-boto.ipynb` is kept in
`legacy/` only because it was one of the two notebooks originally handed
over, not because it did anything.
