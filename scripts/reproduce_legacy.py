"""Faithfully reproduces the legacy notebook's methodology on the same
raw data, as a documented "before" baseline for the rebuild's writeup.

This intentionally keeps every flaw found in ml_code.ipynb:
  - random_state train_test_split() shuffles time-series rows (no
    chronological separation between train and test)
  - weather features and demand target share the same timestamp (no
    forecast horizon -- this is same-time nowcasting dressed up as if
    it were predictive)
  - demand is averaged across ALL 76 EIA respondent codes -- individual
    balancing authorities, EIA's own regional rollups, and the US48
    national total all mixed into one groupby-mean per timestamp
  - no baseline model to contextualize the reported R²

Do not use this as a template for anything else in this repo -- it
exists solely to reproduce the flawed numbers being corrected. See
README.md for the before/after comparison against run_pipeline.py.

Usage:
    python scripts/reproduce_legacy.py [--config config/config.yaml]
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.metrics import compute_metrics
from src.features.weather_features import compute_weather_features
from src.ingestion.era5 import load_era5_conus_mean
from src.models.ml_models import fit_predict


def load_legacy_demand_series(csv_path: str) -> pd.Series:
    """Reproduces preprocess_dataset_energy()'s unfiltered
    groupby(['period', 'type']).mean() -- averaging demand across every
    respondent code in the file, not just one balancing authority or
    the national total.
    """
    df = pd.read_csv(csv_path, usecols=["period", "type", "value"])
    df = df[df["type"] == "D"]
    series = df.groupby("period")["value"].mean()
    series.index = pd.to_datetime(series.index)
    series = series.sort_index().dropna()
    series.index.name = "time"
    series.name = "demand"
    return series


def main(config_path: str):
    cfg = yaml.safe_load(open(config_path))
    tables_dir = Path(cfg["output"]["tables_dir"])
    tables_dir.mkdir(parents=True, exist_ok=True)

    print("Loading raw data (legacy methodology)...")
    weather_raw = load_era5_conus_mean(cfg["data"]["era5_glob"])
    demand = load_legacy_demand_series(cfg["data"]["eia_csv"])

    # Same rolling/lag/diff feature logic as the legacy add_lag_mean(),
    # but NOT shifted by a forecast horizon -- features and target share
    # the same timestamp, exactly as in ml_code.ipynb.
    weather_feat = compute_weather_features(
        weather_raw, cfg["features"]["weather_roll_windows"], cfg["features"]["weather_lag_steps"]
    )

    common_index = weather_feat.index.intersection(demand.index)
    X = weather_feat.loc[common_index].dropna(how="any")
    y = demand.reindex(X.index)
    print(f"  {len(X)} rows, {X.shape[1]} features (weather only, no target lags, no horizon shift)")

    # Legacy test_train_data(): random_state=0, test_size=0.2, shuffle=True (default).
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    print(f"  random split: {len(X_train)} train / {len(X_test)} test (NOT chronological)")

    predictions = {}
    for model_name in ["linear_regression", "random_forest", "xgboost"]:
        params = cfg["models"].get(model_name, {})
        predictions[model_name] = fit_predict(model_name, params, X_train, y_train, X_test)

    metrics = compute_metrics(y_test, predictions)
    metrics = metrics.sort_values("RMSE")
    print("\n=== Legacy-methodology metrics (random split, no horizon, mixed-respondent demand) ===")
    print(metrics)
    print("\nThese numbers are NOT valid evidence of forecasting skill -- see README.md.")

    metrics.to_csv(tables_dir / "legacy_reproduction_metrics.csv")
    print(f"\nSaved to {tables_dir / 'legacy_reproduction_metrics.csv'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()
    main(args.config)
