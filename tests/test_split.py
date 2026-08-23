import pandas as pd
import pytest

from src.models.split import chronological_split


@pytest.fixture
def linear_dataset():
    idx = pd.date_range("2025-01-01", periods=24 * 100, freq="h")
    X = pd.DataFrame({"a": range(len(idx))}, index=idx)
    y = pd.Series(range(len(idx)), index=idx)
    benchmarks = pd.DataFrame({"persistence_24h": range(len(idx))}, index=idx)
    return X, y, benchmarks


def test_splits_are_chronological_and_non_overlapping(linear_dataset):
    X, y, benchmarks = linear_dataset
    splits = chronological_split(X, y, benchmarks, train_end="2025-02-01", val_end="2025-02-15")

    assert splits["train"]["X"].index.max() <= pd.Timestamp("2025-02-01")
    assert splits["val"]["X"].index.min() > pd.Timestamp("2025-02-01")
    assert splits["val"]["X"].index.max() <= pd.Timestamp("2025-02-15")
    assert splits["test"]["X"].index.min() > pd.Timestamp("2025-02-15")

    train_idx, val_idx, test_idx = (splits[s]["X"].index for s in ("train", "val", "test"))
    assert len(train_idx.intersection(val_idx)) == 0
    assert len(val_idx.intersection(test_idx)) == 0
    assert len(train_idx.intersection(test_idx)) == 0
    assert len(train_idx) + len(val_idx) + len(test_idx) == len(X)


def test_split_keeps_X_y_benchmarks_aligned(linear_dataset):
    X, y, benchmarks = linear_dataset
    splits = chronological_split(X, y, benchmarks, train_end="2025-02-01", val_end="2025-02-15")
    for name in ("train", "val", "test"):
        assert splits[name]["X"].index.equals(splits[name]["y"].index)
        assert splits[name]["X"].index.equals(splits[name]["benchmarks"].index)


def test_val_end_after_data_range_raises(linear_dataset):
    X, y, benchmarks = linear_dataset
    with pytest.raises(ValueError):
        chronological_split(X, y, benchmarks, train_end="2025-01-01", val_end="2030-01-01")
