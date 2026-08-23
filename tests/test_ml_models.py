import numpy as np
import pandas as pd
import pytest

from src.models.ml_models import build_model, fit_predict


@pytest.fixture
def toy_regression():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.normal(size=200), "b": rng.normal(size=200)})
    y = 3 * X["a"] - 2 * X["b"] + rng.normal(scale=0.01, size=200)
    return X.iloc[:150], X.iloc[150:], y.iloc[:150], y.iloc[150:]


def test_build_model_unknown_name_raises():
    with pytest.raises(ValueError):
        build_model("not_a_model", {})


@pytest.mark.parametrize("model_name", ["linear_regression", "random_forest", "xgboost"])
def test_fit_predict_shape_and_reasonable_fit(model_name, toy_regression):
    X_train, X_test, y_train, y_test = toy_regression
    preds = fit_predict(model_name, {}, X_train, y_train, X_test)
    assert len(preds) == len(X_test)
    # near-linear synthetic data should be fit well by all three models
    assert np.corrcoef(preds, y_test)[0, 1] > 0.9
