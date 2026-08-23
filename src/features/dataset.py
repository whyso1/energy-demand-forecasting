"""Assembles the leak-safe modeling dataset.

Every row is indexed by the TARGET timestamp t (the hour being
predicted). horizon_hours defines issuance time = t - horizon_hours,
i.e. how far ahead the forecast is made. Two independent mechanisms
keep information from after issuance time out of X:

  1. Weather features are built entirely from data at or before a given
     "source time" (features/weather_features.py only looks backward),
     then the whole feature block is shifted forward by horizon_hours so
     what was known at source_time now sits on the row for target time
     source_time + horizon_hours.
  2. Target (demand) lag/rolling features are built directly against
     target time t, but every lag/window is asserted to be >= horizon_hours
     (features/target_features.py raises if not).

y is the actual demand value at t. Everything in X for a given row is
therefore knowable at t - horizon_hours, which is what tests/test_no_leakage.py
verifies mechanically rather than just by convention.
"""
import numpy as np
import pandas as pd

from src.features.target_features import compute_target_features
from src.features.time_features import compute_time_features
from src.features.weather_features import compute_weather_features
from src.ingestion.eia import load_eia_series
from src.ingestion.era5 import load_era5_conus_mean


def build_dataset(cfg: dict) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Returns (X, y, benchmarks) all aligned on the same target-time index.

    benchmarks holds non-ML reference predictions (persistence, seasonal
    naive, EIA's own day-ahead forecast) for the same rows, so models and
    baselines are scored on an identical test set.
    """
    horizon = cfg["forecast"]["horizon_hours"]
    respondent = cfg["data"]["respondent"]

    weather_raw = load_era5_conus_mean(cfg["data"]["era5_glob"])
    demand = load_eia_series(cfg["data"]["eia_csv"], respondent, cfg["data"]["demand_type"])
    day_ahead = load_eia_series(cfg["data"]["eia_csv"], respondent, cfg["data"]["forecast_type"])

    weather_feat_source = compute_weather_features(
        weather_raw,
        cfg["features"]["weather_roll_windows"],
        cfg["features"]["weather_lag_steps"],
    )
    weather_feat_target = weather_feat_source.shift(horizon, freq="h")

    target_feat = compute_target_features(
        demand,
        cfg["features"]["target_lag_hours"],
        cfg["features"]["target_roll_windows_hours"],
        horizon,
    )

    common_index = weather_feat_target.index.intersection(target_feat.index).intersection(demand.index)
    calendar_feat = compute_time_features(common_index)

    X = pd.concat(
        [weather_feat_target.loc[common_index], target_feat.loc[common_index], calendar_feat],
        axis=1,
    )
    X = X.dropna(how="any")

    y = demand.reindex(X.index)
    y.name = "demand"

    benchmarks = pd.DataFrame(index=X.index)
    for lag in cfg["baselines"]["persistence_lags_hours"]:
        benchmarks[f"persistence_{lag}h"] = demand.shift(lag).reindex(X.index)

    n_cycles = cfg["baselines"]["seasonal_naive"]["lookback_cycles"]
    weekly_lags = [168 * k for k in range(1, n_cycles + 1)]
    seasonal = pd.concat([demand.shift(lag) for lag in weekly_lags], axis=1).mean(axis=1)
    benchmarks["seasonal_naive"] = seasonal.reindex(X.index)

    benchmarks["eia_day_ahead"] = day_ahead.reindex(X.index)

    return X, y, benchmarks
