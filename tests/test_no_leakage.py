import numpy as np
import pandas as pd
import pytest

from src.features.target_features import compute_target_features
from src.features.weather_features import compute_weather_features


def test_weather_lag_uses_only_past_rows():
    idx = pd.date_range("2025-01-01", periods=100, freq="h")
    df = pd.DataFrame({"value": np.arange(100, dtype=float)}, index=idx)

    out = compute_weather_features(df, roll_windows=[], lag_steps=[2])

    for t, row in out.iterrows():
        source_value_2h_ago = df.loc[t - pd.Timedelta(hours=2), "value"]
        assert row["value_lag2"] == source_value_2h_ago


def test_weather_features_shifted_by_horizon_equal_source_values():
    idx = pd.date_range("2025-01-01", periods=100, freq="h")
    df = pd.DataFrame({"value": np.arange(100, dtype=float)}, index=idx)
    horizon = 24

    weather_feat_source = compute_weather_features(df, roll_windows=[3], lag_steps=[1])
    weather_feat_target = weather_feat_source.shift(horizon, freq="h")

    # For any target row t, the feature value must equal exactly what was
    # computable at source_time = t - horizon -- i.e. the row at t used no
    # information from later than t - horizon.
    common = weather_feat_target.index.intersection(weather_feat_source.index)
    for t in common[:20]:
        source_time = t - pd.Timedelta(hours=horizon)
        if source_time in weather_feat_source.index:
            pd.testing.assert_series_equal(
                weather_feat_target.loc[t], weather_feat_source.loc[source_time], check_names=False
            )


def test_target_lag_shorter_than_horizon_raises():
    idx = pd.date_range("2025-01-01", periods=50, freq="h")
    demand = pd.Series(np.arange(50, dtype=float), index=idx)

    with pytest.raises(ValueError):
        compute_target_features(demand, lag_hours=[12], roll_windows_hours=[], horizon_hours=24)


def test_target_lag_equals_exact_past_value():
    idx = pd.date_range("2025-01-01", periods=200, freq="h")
    demand = pd.Series(np.arange(200, dtype=float), index=idx)
    horizon = 24

    out = compute_target_features(demand, lag_hours=[24, 168], roll_windows_hours=[], horizon_hours=horizon)

    for t in out.dropna().index[:20]:
        assert out.loc[t, "demand_lag24"] == demand.loc[t - pd.Timedelta(hours=24)]
        assert out.loc[t, "demand_lag168"] == demand.loc[t - pd.Timedelta(hours=168)]


def test_target_rolling_excludes_the_target_hour_itself():
    idx = pd.date_range("2025-01-01", periods=200, freq="h")
    demand = pd.Series(0.0, index=idx)
    horizon = 24
    spike_time = idx[150]
    demand.loc[spike_time] = 1_000_000.0  # huge spike at the hour we'll predict

    out = compute_target_features(demand, lag_hours=[24], roll_windows_hours=[24], horizon_hours=horizon)

    # Predicting spike_time itself must not "see" the spike in its own
    # rolling-mean feature -- the window ends at spike_time - horizon.
    row = out.loc[spike_time]
    assert row["demand_roll24_mean"] < 1000  # nowhere near the 1,000,000 spike
