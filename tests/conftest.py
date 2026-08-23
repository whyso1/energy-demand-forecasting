import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def hourly_index():
    return pd.date_range("2025-01-01", periods=24 * 30, freq="h")


@pytest.fixture
def synthetic_weather(hourly_index):
    n = len(hourly_index)
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "t2m": 15 + 10 * np.sin(np.arange(n) * 2 * np.pi / 24) + rng.normal(0, 1, n),
            "u10": rng.normal(0, 3, n),
            "v10": rng.normal(0, 3, n),
        },
        index=hourly_index,
    )


@pytest.fixture
def synthetic_demand(hourly_index):
    n = len(hourly_index)
    daily = 1000 + 200 * np.sin(np.arange(n) * 2 * np.pi / 24)
    weekly_dow = hourly_index.dayofweek.values
    weekday_boost = np.where(weekly_dow < 5, 100, 0)
    series = pd.Series(daily + weekday_boost, index=hourly_index, name="D")
    series.index.name = "time"
    return series


@pytest.fixture
def synthetic_day_ahead(synthetic_demand):
    rng = np.random.default_rng(1)
    noisy = synthetic_demand + rng.normal(0, 20, len(synthetic_demand))
    noisy.name = "DF"
    return noisy


@pytest.fixture
def base_cfg():
    return {
        "data": {"respondent": "US48", "demand_type": "D", "forecast_type": "DF"},
        "forecast": {"horizon_hours": 24},
        "features": {
            "weather_roll_windows": [3, 6],
            "weather_lag_steps": [1, 2, 3],
            "target_lag_hours": [24, 48, 168],
            "target_roll_windows_hours": [24],
        },
        "baselines": {
            "persistence_lags_hours": [24, 168],
            "seasonal_naive": {"lookback_cycles": 2},
        },
    }
