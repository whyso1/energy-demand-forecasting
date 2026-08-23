"""Corrected, leak-safe energy demand forecasting pipeline.

Chronological split, real baselines, and target-lag features -- see
README.md for the before/after story against scripts/reproduce_legacy.py.

Usage:
    python run_pipeline.py [--config config/config.yaml]
"""
import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

from src.evaluation.metrics import compute_metrics
from src.evaluation.peak_events import evaluate_peak_predictions
from src.features.dataset import build_dataset
from src.models.ml_models import fit_predict
from src.models.split import chronological_split


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main(config_path: str):
    cfg = load_config(config_path)
    tables_dir = Path(cfg["output"]["tables_dir"])
    tables_dir.mkdir(parents=True, exist_ok=True)

    print("Building leak-safe dataset...")
    X, y, benchmarks = build_dataset(cfg)
    print(f"  {len(X)} rows, {X.shape[1]} features, respondent={cfg['data']['respondent']}, "
          f"horizon={cfg['forecast']['horizon_hours']}h")

    splits = chronological_split(X, y, benchmarks, cfg["split"]["train_end"], cfg["split"]["val_end"])
    for name in ("train", "val", "test"):
        s = splits[name]
        print(f"  {name}: {len(s['X'])} rows, {s['X'].index.min()} -> {s['X'].index.max()}")

    test = splits["test"]
    predictions = {}

    print("Evaluating baselines on test set...")
    for col in test["benchmarks"].columns:
        predictions[col] = test["benchmarks"][col]

    print("Fitting ML models on train set, predicting on test set...")
    for model_name in ["linear_regression", "random_forest", "xgboost"]:
        params = cfg["models"].get(model_name, {})
        preds = fit_predict(model_name, params, splits["train"]["X"], splits["train"]["y"], test["X"])
        predictions[model_name] = pd.Series(preds, index=test["X"].index)

    # Baselines can have NaN at the start of the test window (e.g. seasonal_naive
    # needs 4 weeks of prior history); score every model/baseline on the same
    # rows so the comparison is apples-to-apples.
    valid_index = test["y"].index
    for preds in predictions.values():
        valid_index = valid_index.intersection(preds.dropna().index)
    y_eval = test["y"].loc[valid_index]
    predictions_eval = {name: preds.loc[valid_index] for name, preds in predictions.items()}
    print(f"  scoring on {len(valid_index)} rows common to all models/baselines "
          f"({test['y'].index.min()} -> {test['y'].index.max()})")

    metrics = compute_metrics(y_eval, predictions_eval)
    metrics = metrics.sort_values("RMSE")
    print("\n=== Test-set metrics (chronological split, all models/baselines on identical rows) ===")
    print(metrics)
    metrics.to_csv(tables_dir / "corrected_pipeline_metrics.csv")

    ml_only_preds = {k: v for k, v in predictions_eval.items() if k in ("linear_regression", "random_forest", "xgboost")}
    peak_top_n = cfg["peak_evaluation"]["top_n"]
    peak_hit_window = cfg["peak_evaluation"]["hit_window_hours"]
    _, peak_summary = evaluate_peak_predictions(y_eval, ml_only_preds, top_n=peak_top_n, hit_window_hours=peak_hit_window)
    print(f"\n=== Peak-event evaluation (top {peak_top_n} demand peaks, {peak_hit_window}h hit window) ===")
    print(peak_summary)
    peak_summary.to_csv(tables_dir / "corrected_pipeline_peak_summary.csv", index=False)

    with open(tables_dir / "run_metadata.json", "w") as f:
        json.dump(
            {
                "respondent": cfg["data"]["respondent"],
                "horizon_hours": cfg["forecast"]["horizon_hours"],
                "n_features": X.shape[1],
                "train_rows": len(splits["train"]["X"]),
                "val_rows": len(splits["val"]["X"]),
                "test_rows_scored": len(valid_index),
                "test_range": [str(y_eval.index.min()), str(y_eval.index.max())],
            },
            f,
            indent=2,
        )

    print(f"\nSaved results to {tables_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()
    main(args.config)
