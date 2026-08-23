"""Chronological train/val/test split.

Replaces the legacy sklearn.train_test_split(..., shuffle=True default),
which randomly shuffled time-series rows across train/test -- letting
the model train on data adjacent in time to what it was tested on and
invalidating any of the reported metrics as evidence of forecasting
skill. A fixed chronological cutoff is the simplest correct split; it
can be swapped for rolling-origin CV later without touching the rest of
the pipeline.
"""
import pandas as pd


def chronological_split(
    X: pd.DataFrame, y: pd.Series, benchmarks: pd.DataFrame, train_end: str, val_end: str
) -> dict:
    train_end = pd.Timestamp(train_end)
    val_end = pd.Timestamp(val_end)

    train_mask = X.index <= train_end
    val_mask = (X.index > train_end) & (X.index <= val_end)
    test_mask = X.index > val_end

    if not test_mask.any():
        raise ValueError(f"No rows after val_end={val_end}; check split cutoffs against data range.")

    splits = {}
    for name, mask in [("train", train_mask), ("val", val_mask), ("test", test_mask)]:
        splits[name] = {
            "X": X.loc[mask],
            "y": y.loc[mask],
            "benchmarks": benchmarks.loc[mask],
        }
    return splits
