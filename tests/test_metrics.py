import numpy as np
import pandas as pd

from src.evaluation.metrics import compute_metrics


def test_perfect_predictions_score_r2_one():
    y_true = pd.Series([1.0, 2.0, 3.0, 4.0])
    metrics = compute_metrics(y_true, {"perfect": y_true.copy()})
    assert np.isclose(metrics.loc["perfect", "R2"], 1.0)
    assert np.isclose(metrics.loc["perfect", "MSE"], 0.0)


def test_metrics_use_the_passed_y_true_not_a_global(monkeypatch):
    """Regression guard for the legacy bug where model_eval() read a
    module-global `y_test` instead of a parameter -- compute_metrics
    must only ever use the y_true it's given.
    """
    y_true_a = pd.Series([1.0, 2.0, 3.0])
    y_true_b = pd.Series([10.0, 20.0, 30.0])
    preds = pd.Series([1.0, 2.0, 3.0])

    metrics_a = compute_metrics(y_true_a, {"m": preds})
    metrics_b = compute_metrics(y_true_b, {"m": preds})

    assert not np.isclose(metrics_a.loc["m", "MSE"], metrics_b.loc["m", "MSE"])
    assert np.isclose(metrics_a.loc["m", "R2"], 1.0)
