# Energy Demand Forecasting Pipeline

A leak-safe, chronologically-validated rebuild of a legacy demand-forecasting
notebook (`legacy/ml_code.ipynb`). Predicts US48 (EIA's Lower-48 aggregate)
hourly electricity demand 24 hours ahead from ERA5 reanalysis weather,
benchmarked against real baselines -- including EIA's own published
day-ahead demand forecast.

## TL;DR: before vs. after

| | Legacy methodology | Corrected methodology |
|---|---|---|
| Split | Random 80/20 shuffle | Chronological (train -> val -> test, in time order) |
| Horizon | None (same timestamp) | 24h ahead, enforced by construction |
| Demand series | Mean across all 76 EIA respondent codes (BAs + regional rollups + national total, mixed) | `US48` only (EIA's own Lower-48 total) |
| Baselines | None | Persistence (t-24h, t-168h), seasonal-naive, EIA's own day-ahead forecast |
| Target lags | None | Yes, `>= horizon` enforced |

**Legacy (flawed) test-set R²** (`scripts/reproduce_legacy.py`, `results/tables/legacy_reproduction_metrics.csv`):

| Model | R2 | RMSE |
|---|---|---|
| XGBoost | 0.982 | 509 |
| Random Forest | 0.963 | 718 |
| Linear Regression | 0.786 | 1735 |

**Corrected (leak-safe) test-set metrics**, 2025-10-01 -> 2025-11-14
(`run_pipeline.py`, `results/tables/corrected_pipeline_metrics.csv`):

| Model | R2 | MAPE | RMSE |
|---|---|---|---|
| Random Forest | 0.910 | 2.19% | 13,489 |
| XGBoost | 0.909 | 2.28% | 13,562 |
| Linear Regression | 0.862 | 3.07% | 16,744 |
| Persistence (t-24h) | 0.821 | 3.03% | 19,045 |
| Persistence (t-168h) | 0.739 | 3.87% | 23,024 |
| **EIA's own day-ahead forecast** | 0.738 | **2.14%** | 23,045 |
| Seasonal-naive | 0.318 | 6.56% | 37,224 |

The legacy R²=0.98-0.99 numbers were never evidence of forecasting skill --
they came from training and testing on randomly interleaved timestamps
(so the model could effectively memorize nearby hours) with weather and
demand read at the *same* instant (so "prediction" was closer to
interpolation than forecasting). Once both are fixed, R² drops to a much
more mundane 0.86-0.91 -- a real number, not a leakage artifact.

The honest headline, though, is the comparison to `eia_day_ahead`: Random
Forest and XGBoost beat every naive baseline by a wide margin (R² 0.91 vs.
0.82 persistence vs. 0.32 seasonal-naive) and edge out EIA's own published
forecast on R²/RMSE, but EIA's forecast still has a *slightly* lower MAPE
(2.14% vs. 2.19%) and much better MAE (9,275 vs. 9,973) than the best model
here. R² is sensitive to the variance of the test window (a low-swing test
period flatters MAPE-good, high-error-relative-to-that-variance models less
generously), which is why MAPE/MAE and R² tell slightly different stories
for `eia_day_ahead` -- see `src/evaluation/metrics.py`. Net: the ML models
here are competitive with a real, professionally-produced operational
forecast, not obviously better than it. That is a much more defensible
claim than the legacy R²=0.992.

## Design decisions

These were open questions before this rebuild; verified against the actual
data and confirmed:

- **Forecast horizon: 24 hours ahead.** Matches EIA's own day-ahead demand
  forecast (`type=DF`), giving a real operational baseline to compare
  against, rather than an arbitrary horizon with nothing to benchmark it.
- **Weather information cutoff: lagged ERA5 only.** Every weather feature
  used to predict target time `t` is built from ERA5 data at or before
  `t - 24h`, never later. Note this is a documented simplification: the
  `.nc` files are actually **ERA5T** (Copernicus's preliminary/near-real-time
  product, not final ERA5) -- confirmed from the actual acquisition script
  (`~/Desktop/energy/save_data.ipynb`, which explicitly requests ERA5T via
  the CDS API), correcting an earlier version of this document that assumed
  `expver=0001` meant final ERA5; that tag doesn't reliably distinguish the
  two for this CDS endpoint. ERA5T typically publishes within a few days of
  real time but isn't final -- ECMWF later reprocesses it into consolidated
  ERA5 (with possible small revisions) roughly 2-3 months after the fact.
  A true real-time system would need to either accept ERA5T's preliminary
  status and revision risk, or use actual forecast weather (e.g. GFS/HRRR)
  instead of reanalysis. Treat this pipeline's numbers as an upper bound on
  what's achievable with lagged-reanalysis-only weather, not a deployable
  forecast. The same caveat applies to demand
  actuals: `demand_lag{horizon}` assumes EIA's `D` (actuals) value at exactly
  `t - horizon` is available at issuance time `t - horizon`. Unlike `DF`
  (a forecast, genuinely known in advance by construction), EIA's hourly
  actuals can carry their own reporting/revision lag in practice -- this
  pipeline doesn't model that, same as it doesn't model ERA5's real
  publication lag.
- **Geographic scope: `respondent == "US48"`.** EIA's own Lower-48
  aggregate, which pairs with the CONUS-mean ERA5 weather already being
  computed. See "Bonus finding" below for why this isn't just an arbitrary
  choice.
- **EIA `type=D` semantics:** confirmed directly from the CSV's own
  `type-name` column: `D` = `"Demand"`, `DF` = `"Day-ahead demand forecast"`.

## Bonus finding: the legacy demand series was double-counted

`preprocess_dataset_energy()` in the legacy notebook did
`df.groupby(["period", "type"]).mean()` with no filter on `respondent`.
The EIA CSV has 76 respondent codes: individual balancing authorities
(`CISO`, `ERCO`, `PJM`, ...), EIA's own regional rollups (`CAL`, `CENT`,
`MIDW`, `NE`, `SE`, ...), *and* `US48` (the national total) -- all averaged
together into one number per timestamp. That's not "US demand," it's an
unweighted mean of a national total, several regional sub-totals, and
dozens of individual utility areas, which is not a physically meaningful
quantity. `src/ingestion/eia.py` fixes this by filtering to a single
`respondent` before aggregating (default: `US48`).

## Repository layout

```
config/config.yaml          # all paths, horizon, split cutoffs, model hyperparams
src/
  ingestion/                # era5.py, eia.py -- load + clean raw data
  features/                 # weather_features.py, target_features.py, time_features.py,
                             # dataset.py (assembles the leak-safe X/y/benchmarks)
  models/                   # split.py (chronological split), ml_models.py
  evaluation/               # metrics.py, peak_events.py
tests/                      # no-leakage, alignment, schema, split, model tests
scripts/reproduce_legacy.py # faithfully reproduces the legacy (flawed) result
run_pipeline.py             # the corrected end-to-end pipeline
legacy/                     # original ml_code.ipynb, preserved as-is; era5-s3-via-boto.ipynb
                             # is a stale, never-actually-used AWS example, kept only because
                             # it was one of the two notebooks originally handed over --
                             # see AUDIT.md section 9 for what the real ERA5T pull looks like
data/raw/                   # symlinks to the raw ERA5 .nc files and eia_load_data.csv
                             # (kept in ~/Desktop/energy/, not duplicated -- see below)
results/tables/             # output CSVs from both scripts
```

`data/raw/*` are symlinks, not copies -- the underlying ERA5 files
(~2.1 GB across 11 months) and the EIA CSV (~185 MB) live in
`~/Desktop/energy/` and are referenced rather than duplicated. If that
directory moves, recreate the symlinks or point `config/config.yaml`'s
`data.era5_glob` / `data.eia_csv` at the new location.

## What was preserved vs. rebuilt

**Preserved (relocated, lightly cleaned):**
- `preprocess_dataset_era5()` -> `src/ingestion/era5.py` (CONUS clip, K->C,
  wind speed, spatial mean). Unused `cartopy` import dropped.
- Rolling/lag/diff feature framework (`add_lag_mean()`) ->
  `src/features/weather_features.py`, now operating on a complete hourly
  grid so an "N-row window" always means "N hours" even if the source data
  has gaps (the legacy version's row-count windows would silently mean
  something else if data was missing).
- Time features (hour/dayofweek/month/weekend/holiday) ->
  `src/features/time_features.py`.
- Peak-event evaluation concept (`evaluate_peak_predictions()`) ->
  `src/evaluation/peak_events.py`, computation split from plotting so it's
  unit-testable; `y_test` is now an explicit argument.

**Rebuilt from scratch:**
- Chronological train/val/test split (`src/models/split.py`) replacing
  `train_test_split(..., random_state=0)`. **Note:** `run_pipeline.py`
  currently only reads `splits["train"]` and `splits["test"]` --
  `splits["val"]` is produced but not yet used for anything (no early
  stopping, no hyperparameter selection). It's reserved for that future
  work, not silently doing something it isn't; see "Known limitations"
  below.
- Baselines: persistence (t-24h, t-168h), seasonal-naive (mean of the same
  hour-of-week over the prior N weeks), and EIA's own day-ahead forecast --
  computed in `src/features/dataset.py`'s `benchmarks` output, evaluated
  before any ML model.
- Target lag/rolling features (`src/features/target_features.py`), which
  didn't exist at all in the legacy notebook -- only weather had lag
  features.
- `config/config.yaml` externalizing every path and hyperparameter that was
  previously hard-coded (`eta_2025_*.nc`, `eia_load_data.csv`, split dates,
  model params).
- `src/evaluation/metrics.py`'s `compute_metrics()` takes `y_true`
  explicitly; the legacy `model_eval()` read a module-global `y_test`,
  which only worked by accident because every call site happened to share
  that global.
- The unused `n_estimators = 1000` assignment (set in a loop, never passed
  into `ml_model()`) is gone; `n_estimators` is now a normal config value.
- Tests (`tests/`): no-leakage checks (every weather/target lag feature is
  traced back to an exact past timestamp), X/y/benchmark index alignment,
  expected-columns schema, target column absent from X, chronological-split
  correctness.

## Running it

```bash
pip install -r requirements.txt
python -m pytest                      # unit tests, ~5s, no data files needed
python scripts/reproduce_legacy.py    # "before": legacy methodology, real data
python run_pipeline.py                # "after": corrected methodology, real data
```

Both scripts read `config/config.yaml` by default (`--config` to override)
and write to `results/tables/`.

## Known limitations / next steps

- **`val` split is unused.** `chronological_split()` produces train/val/test,
  but `run_pipeline.py` only trains on `train` and scores on `test` -- `val`
  isn't wired into anything yet (e.g. picking between a couple of
  hyperparameter configs per model, or early stopping for XGBoost). Treat it
  as reserved, not as evidence that any model selection happened.
- **Weather latency is idealized.** As noted above, this uses ERA5T
  (preliminary, itself subject to later revision when consolidated into
  final ERA5) as if it were available exactly at `t-24h`. A deployable
  version needs actual forecast weather (GFS/HRRR), since even ERA5T isn't
  produced fast enough or guaranteed stable enough for real-time use as-is.
- **Fixed-cutoff split, not rolling-origin.** One train/val/test boundary
  rather than walk-forward cross-validation; reasonable for a first
  correct baseline, but a single test window (here, six autumn weeks) is a
  narrow sample of demand regimes (no summer AC peak, no winter heating
  peak in the test set).
- **US48 is a broad aggregate.** CONUS-mean weather is a coarse proxy for
  demand drivers that are regionally concentrated (e.g. a Texas heat wave
  moves ERCOT demand far more than the CONUS mean temperature). A
  single-BA version (e.g. `CISO`, `ERCO`) with weather clipped to that BA's
  footprint would have tighter physical coupling, at the cost of losing
  the clean `US48` <-> CONUS-mean pairing.

## License

MIT -- see [LICENSE](LICENSE).
