"""Calendar features of the target timestamp.

These are legitimate at any horizon: the calendar structure of a future
timestamp (hour, day of week, month, weekend, holiday) is fully known in
advance, so computing them from the target time itself is not leakage.
"""
import holidays
import pandas as pd


def compute_time_features(target_index: pd.DatetimeIndex) -> pd.DataFrame:
    us_holidays = holidays.US()
    return pd.DataFrame(
        {
            "hour": target_index.hour,
            "dayofweek": target_index.dayofweek,
            "month": target_index.month,
            "weekend": target_index.dayofweek >= 5,
            "holiday": [d in us_holidays for d in target_index.date],
        },
        index=target_index,
    )
