"""Rolling/lag/diff features for meteorology, computed on a complete
hourly grid so that an integer window of N always means N hours (the
legacy add_lag_mean() used positional rolling/shift, which silently
means something different from "N hours" if the source data has gaps).

Every column produced here at row `source_time` depends only on weather
data at `source_time` and earlier -- rolling/shift/diff all look
backward. That makes this dataframe, before any horizon shift, "known
as of source_time". src/features/dataset.py shifts it forward by the
forecast horizon to align it with the target timestamp it's allowed to
be used for.
"""
import numpy as np
import pandas as pd


def compute_weather_features(
    weather: pd.DataFrame, roll_windows: list[int], lag_steps: list[int]
) -> pd.DataFrame:
    full_index = pd.date_range(weather.index.min(), weather.index.max(), freq="h")
    df = weather.reindex(full_index)
    df.index.name = "time"

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    new_cols = {}

    for col in numeric_cols:
        for w in roll_windows:
            new_cols[f"{col}_roll{w}_mean"] = df[col].rolling(w, min_periods=1).mean()
            new_cols[f"{col}_roll{w}_std"] = df[col].rolling(w, min_periods=1).std()
        for lag in lag_steps:
            new_cols[f"{col}_lag{lag}"] = df[col].shift(lag)
        diff1 = df[col].diff()
        new_cols[f"{col}_diff1"] = diff1
        new_cols[f"{col}_diff2"] = diff1.diff()

    out = pd.concat([df[numeric_cols], pd.DataFrame(new_cols, index=df.index)], axis=1)

    max_window = max(roll_windows + lag_steps + [2]) if (roll_windows or lag_steps) else 2
    return out.iloc[max_window:]
