"""Regression metrics table, ported from the legacy model_eval().

The legacy version read `y_test` from the notebook's global scope
instead of taking it as a parameter -- harmless only by accident, since
every call in the notebook happened to use the same global. Here it's
an explicit argument, so this function gives the same answer regardless
of what else is in scope when it's called.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score


def compute_metrics(y_true: pd.Series, predictions: dict[str, pd.Series]) -> pd.DataFrame:
    """predictions: {name: array-like of predictions aligned with y_true}.

    Includes MAPE alongside R²: R² is normalized by the variance of
    y_true within whatever window it's computed on, so it can look
    unimpressive for an accurate model if the test window happens to be
    low-variance (e.g. EIA's own day-ahead forecast: ~2% MAPE, well
    within its real-world published accuracy, but R2 ~0.74 on a fall
    test window with comparatively little swing). MAPE is scale-relative
    instead and doesn't have that failure mode.
    """
    results = {}
    y_var = np.var(y_true)
    for name, preds in predictions.items():
        mse = mean_squared_error(y_true, preds)
        results[name] = {
            "MSE": mse,
            "RMSE": np.sqrt(mse),
            "MAE": mean_absolute_error(y_true, preds),
            "MAPE": mean_absolute_percentage_error(y_true, preds),
            "R2": r2_score(y_true, preds),
            "Normalized MSE": mse / y_var,
        }
    return pd.DataFrame(results).T
