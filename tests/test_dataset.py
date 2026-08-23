import src.features.dataset as dataset_module


def test_build_dataset_alignment_and_schema(
    monkeypatch, base_cfg, synthetic_weather, synthetic_demand, synthetic_day_ahead
):
    def fake_era5(glob_pattern):
        return synthetic_weather

    def fake_eia(csv_path, respondent, type_code):
        return synthetic_demand if type_code == "D" else synthetic_day_ahead

    monkeypatch.setattr(dataset_module, "load_era5_conus_mean", fake_era5)
    monkeypatch.setattr(dataset_module, "load_eia_series", fake_eia)

    cfg = dict(base_cfg)
    cfg["data"] = {**cfg["data"], "era5_glob": "unused", "eia_csv": "unused"}

    X, y, benchmarks = dataset_module.build_dataset(cfg)

    assert len(X) > 0, "dataset ended up empty -- check synthetic fixture length vs. lag/horizon config"

    # X/y/benchmarks must be perfectly aligned on the same target-time index
    assert X.index.equals(y.index)
    assert X.index.equals(benchmarks.index)

    # the label must never appear as a feature
    assert "demand" not in X.columns
    assert "D" not in X.columns

    # expected feature families are present
    assert any(c.startswith("t2m_lag") for c in X.columns)
    assert any(c.startswith("demand_lag") for c in X.columns)
    for calendar_col in ["hour", "dayofweek", "month", "weekend", "holiday"]:
        assert calendar_col in X.columns

    # benchmark columns present and distinct from ML features
    for bench_col in ["persistence_24h", "persistence_168h", "seasonal_naive", "eia_day_ahead"]:
        assert bench_col in benchmarks.columns

    assert not X.isna().any().any(), "dropna(how='any') should have removed all remaining NaN rows"


def test_no_row_uses_demand_from_after_its_own_issuance_time(
    monkeypatch, base_cfg, synthetic_weather, synthetic_demand, synthetic_day_ahead
):
    """Every demand_lag* column at target row t must equal the actual
    demand value at t - lag -- i.e. traceable to a specific past
    timestamp, never to t itself or later.
    """
    monkeypatch.setattr(dataset_module, "load_era5_conus_mean", lambda g: synthetic_weather)
    monkeypatch.setattr(
        dataset_module, "load_eia_series", lambda p, r, t: synthetic_demand if t == "D" else synthetic_day_ahead
    )

    cfg = dict(base_cfg)
    cfg["data"] = {**cfg["data"], "era5_glob": "unused", "eia_csv": "unused"}
    X, y, benchmarks = dataset_module.build_dataset(cfg)

    import pandas as pd

    for lag in cfg["features"]["target_lag_hours"]:
        col = f"demand_lag{lag}"
        sample = X.index[:10]
        for t in sample:
            expected = synthetic_demand.get(t - pd.Timedelta(hours=lag))
            if expected is not None:
                assert X.loc[t, col] == expected
