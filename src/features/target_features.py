"""Lag/rolling features of the target (demand) series itself.

This is the piece the legacy notebook never had: features derived from
the target's own history. Every lag/window here is required to reach at
least `horizon_hours` back from the target timestamp, so a row built for
target time t never uses demand data from later than t - horizon -- the
same instant issuance time t - horizon "knows" everything else by.
"""
import pandas as pd


def compute_target_features(
    demand: pd.Series,
    lag_hours: list[int],
    roll_windows_hours: list[int],
    horizon_hours: int,
) -> pd.DataFrame:
    bad_lags = [h for h in lag_hours if h < horizon_hours]
    if bad_lags:
        raise ValueError(
            f"target_lag_hours {bad_lags} are shorter than the forecast horizon "
            f"({horizon_hours}h) -- these would leak future demand into the features."
        )

    full_index = pd.date_range(demand.index.min(), demand.index.max(), freq="h")
    d = demand.reindex(full_index)
    d.index.name = "time"

    out = pd.DataFrame(index=d.index)
    for h in lag_hours:
        out[f"demand_lag{h}"] = d.shift(h)

    for w in roll_windows_hours:
        # Rolling stat over a window ending at issuance time (t - horizon),
        # then shifted forward by the horizon so it lands on target row t.
        rolling_mean = d.rolling(w, min_periods=1).mean().shift(horizon_hours)
        rolling_std = d.rolling(w, min_periods=1).std().shift(horizon_hours)
        out[f"demand_roll{w}_mean"] = rolling_mean
        out[f"demand_roll{w}_std"] = rolling_std

    return out
